/**
 * Knowledge-plane infrastructure: a Postgres instance with pgvector, its network boundary,
 * and the secret the platform reads its DSN from.
 *
 * Scope is deliberately narrow. This module owns the stateful dependency that the
 * application cannot recreate on its own; the cluster, VPC and node groups belong to the
 * platform team's own modules and are passed in. A Terraform module that provisions
 * everything is a module nobody can adopt.
 */

terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.60"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

locals {
  is_production = var.environment == "prod"

  tags = merge(
    {
      Application = "enterprise-agent-platform"
      Environment = var.environment
      ManagedBy   = "terraform"
      Component   = "knowledge-plane"
    },
    var.tags,
  )
}

# ---------------------------------------------------------------------------------------
# Network boundary
# ---------------------------------------------------------------------------------------

resource "aws_security_group" "postgres" {
  name        = "${var.name}-${var.environment}-postgres"
  description = "Postgres access for the Enterprise Agent Platform knowledge plane"
  vpc_id      = var.vpc_id
  tags        = merge(local.tags, { Name = "${var.name}-${var.environment}-postgres" })

  lifecycle {
    create_before_destroy = true
  }
}

# Ingress is granted per source security group rather than by CIDR, so the rule survives
# subnet changes and reads as "these workloads may connect" rather than "this address range
# may connect".
resource "aws_vpc_security_group_ingress_rule" "postgres" {
  for_each = toset(var.allowed_security_group_ids)

  security_group_id            = aws_security_group.postgres.id
  referenced_security_group_id = each.value
  from_port                    = 5432
  to_port                      = 5432
  ip_protocol                  = "tcp"
  description                  = "Postgres from ${each.value}"
  tags                         = local.tags
}

resource "aws_db_subnet_group" "this" {
  name        = "${var.name}-${var.environment}"
  description = "Private subnets for the ${var.name} knowledge plane"
  subnet_ids  = var.private_subnet_ids
  tags        = local.tags
}

# ---------------------------------------------------------------------------------------
# Parameter group
# ---------------------------------------------------------------------------------------

resource "aws_db_parameter_group" "this" {
  name        = "${var.name}-${var.environment}-pg${split(".", var.postgres_version)[0]}"
  family      = "postgres${split(".", var.postgres_version)[0]}"
  description = "Enterprise Agent Platform knowledge plane"
  tags        = local.tags

  # pgvector must be preloaded before CREATE EXTENSION will work.
  parameter {
    name         = "shared_preload_libraries"
    value        = "pg_stat_statements,vector"
    apply_method = "pending-reboot"
  }

  # TLS is not optional: the DSN carries credentials and the payload carries customer data.
  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  # Log anything slower than a second. HNSW index builds and unindexed vector scans are the
  # two things that go wrong here, and both are visible at this threshold.
  parameter {
    name  = "log_min_duration_statement"
    value = "1000"
  }

  lifecycle {
    create_before_destroy = true
  }
}

# ---------------------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------------------

resource "random_password" "postgres" {
  length  = 40
  special = true
  # Excluded because these characters need escaping inside a DSN, and a password that
  # breaks the connection string is discovered at the worst possible moment.
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "aws_db_instance" "this" {
  identifier     = "${var.name}-${var.environment}"
  engine         = "postgres"
  engine_version = var.postgres_version
  instance_class = var.instance_class

  db_name  = "eap"
  username = "eap_app"
  password = random_password.postgres.result
  port     = 5432

  allocated_storage     = var.allocated_storage_gb
  max_allocated_storage = var.max_allocated_storage_gb
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = var.kms_key_arn

  db_subnet_group_name   = aws_db_subnet_group.this.name
  vpc_security_group_ids = [aws_security_group.postgres.id]
  parameter_group_name   = aws_db_parameter_group.this.name
  publicly_accessible    = false

  multi_az                = local.is_production
  backup_retention_period = local.is_production ? 30 : var.backup_retention_days
  backup_window           = "03:00-04:00"
  maintenance_window      = "sun:04:30-sun:05:30"
  copy_tags_to_snapshot   = true

  deletion_protection       = local.is_production
  skip_final_snapshot       = !local.is_production
  final_snapshot_identifier = local.is_production ? "${var.name}-${var.environment}-final" : null

  auto_minor_version_upgrade      = true
  apply_immediately               = !local.is_production
  performance_insights_enabled    = local.is_production
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]

  tags = local.tags

  lifecycle {
    # Rotation is handled by Secrets Manager, not by Terraform. Without this, every plan
    # after a rotation shows a password diff and reverts the rotated credential.
    ignore_changes = [password]
  }
}

# ---------------------------------------------------------------------------------------
# Secret
# ---------------------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "dsn" {
  name        = "${var.name}/${var.environment}/postgres-dsn"
  description = "Connection string the Enterprise Agent Platform reads as EAP_DATAOPS_POSTGRES_DSN"
  kms_key_id  = var.kms_key_arn

  # Long enough to recover from an accidental delete, short enough that a rotated secret
  # does not linger.
  recovery_window_in_days = local.is_production ? 30 : 7

  tags = local.tags
}

resource "aws_secretsmanager_secret_version" "dsn" {
  secret_id = aws_secretsmanager_secret.dsn.id

  # sslmode=require, matching rds.force_ssl above. A DSN that omits it connects in
  # cleartext wherever the server permits it.
  secret_string = jsonencode({
    EAP_DATAOPS_POSTGRES_DSN = format(
      "postgresql://%s:%s@%s:%d/%s?sslmode=require",
      aws_db_instance.this.username,
      random_password.postgres.result,
      aws_db_instance.this.address,
      aws_db_instance.this.port,
      aws_db_instance.this.db_name,
    )
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}
