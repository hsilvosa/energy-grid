variable "project_name" {
  description = "Short name used to prefix AWS resources."
  type        = string
  default     = "energy-grid"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for the platform."
  type        = string
  default     = "eu-west-1"
}

variable "vpc_cidr" {
  description = "VPC address range."
  type        = string
  default     = "10.42.0.0/16"
}

variable "github_repository" {
  description = "GitHub owner/repository allowed to assume the deployment role."
  type        = string
  default     = "owner/energy-grid"
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget for the small-scale environment."
  type        = number
  default     = 150
}

variable "api_image" {
  description = "Immutable API image URI. Leave empty to provision shared infrastructure only."
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate for the HTTPS listener. Required when api_image is set."
  type        = string
  default     = ""
}

variable "alert_email" {
  description = "Address for AWS Budget notifications."
  type        = string
  default     = ""
}

