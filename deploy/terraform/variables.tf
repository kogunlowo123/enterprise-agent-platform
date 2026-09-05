variable "name" {
  description = "Name prefix for every resource this module creates."
  type        = string
  default     = "eap"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.name))
    error_message = "name must be lowercase alphanumeric with hyphens, 2-21 characters, starting with a letter."
  }
}

variable "environment" {
  description = "Deployment environment. Drives retention, deletion protection and instance sizing."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "vpc_id" {
  description = "VPC the database is placed in."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnets for the database subnet group. At least two, in different AZs."
  type        = list(string)

  validation {
    condition     = length(var.private_subnet_ids) >= 2
    error_message = "at least two subnets in different availability zones are required."
  }
}

variable "allowed_security_group_ids" {
  description = <<-EOT
    Security groups permitted to reach Postgres on 5432 — normally the EKS node group or
    pod security group. An empty list means nothing can connect, which is the correct
    default: the database should not be reachable until something is explicitly allowed.
  EOT
  type        = list(string)
  default     = []
}

variable "postgres_version" {
  description = "Postgres major version. pgvector is available from 15 onward on RDS."
  type        = string
  default     = "16.4"
}

variable "instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.medium"
}

variable "allocated_storage_gb" {
  description = "Initial storage. Autoscaling raises it up to max_allocated_storage_gb."
  type        = number
  default     = 50
}

variable "max_allocated_storage_gb" {
  description = "Storage autoscaling ceiling. Embeddings grow faster than teams expect."
  type        = number
  default     = 500
}

variable "backup_retention_days" {
  description = "Automated backup retention. Overridden to 30 in prod."
  type        = number
  default     = 7
}

variable "kms_key_arn" {
  description = "Customer-managed KMS key for storage and secret encryption. Null uses the AWS-managed key."
  type        = string
  default     = null
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
