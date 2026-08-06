output "lakehouse_bucket" { value = aws_s3_bucket.lakehouse.id }
output "glue_database" { value = aws_glue_catalog_database.lakehouse.name }
output "msk_bootstrap_brokers" { value = aws_msk_serverless_cluster.main.bootstrap_brokers_sasl_iam }
output "emr_serverless_application_id" { value = aws_emrserverless_application.spark.id }
output "ecr_repositories" { value = { for name, repository in aws_ecr_repository.images : name => repository.repository_url } }
output "github_deploy_role_arn" { value = aws_iam_role.github_deploy.arn }
output "api_endpoint" { value = local.runtime_enabled ? "https://${aws_lb.api[0].dns_name}" : null }

