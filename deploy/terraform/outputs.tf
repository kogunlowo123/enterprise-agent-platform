output "postgres_endpoint" {
  description = "Host and port of the knowledge-plane database."
  value       = aws_db_instance.this.endpoint
}

output "postgres_security_group_id" {
  description = "Attach workloads to this, or reference it, to be granted access on 5432."
  value       = aws_security_group.postgres.id
}

output "dsn_secret_arn" {
  description = <<-EOT
    ARN of the Secrets Manager secret holding EAP_DATAOPS_POSTGRES_DSN. Point External
    Secrets or the Secrets Store CSI driver at this; the Helm chart references a Kubernetes
    Secret by name and never templates a credential itself.
  EOT
  value       = aws_secretsmanager_secret.dsn.arn
}

output "dsn_secret_name" {
  description = "Name of the secret, for the External Secrets remoteRef."
  value       = aws_secretsmanager_secret.dsn.name
}

output "bootstrap_sql" {
  description = <<-EOT
    Run once against the new database as a superuser. The application role does not have
    permission to create extensions, and should not: an application that can install
    arbitrary extensions can install one that reads the filesystem.
  EOT
  value       = "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_stat_statements;"
}
