data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name                       = "${var.project_name}-${var.environment}"
  runtime_enabled            = var.api_image != "" && var.certificate_arn != ""
  task_definition_family_arn = "arn:${data.aws_partition.current.partition}:ecs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:task-definition/${local.name}-api"
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = { Name = local.name }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = local.name }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name}-public-${count.index + 1}" }
}

resource "aws_subnet" "data" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = { Name = "${local.name}-data-${count.index + 1}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  route_table_id = aws_route_table.public.id
  subnet_id      = aws_subnet.public[count.index].id
}

resource "aws_security_group" "runtime" {
  name        = "${local.name}-runtime"
  description = "Internal runtime traffic"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "Internal traffic"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "alb" {
  name   = "${local.name}-alb"
  vpc_id = aws_vpc.main.id
  ingress {
    description = "HTTPS"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group_rule" "alb_to_api" {
  type                     = "ingress"
  security_group_id        = aws_security_group.runtime.id
  source_security_group_id = aws_security_group.alb.id
  from_port                = 8000
  to_port                  = 8000
  protocol                 = "tcp"
  description              = "ALB to FastAPI"
}

resource "aws_s3_bucket" "lakehouse" {
  bucket_prefix = "${local.name}-lakehouse-"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "lakehouse" {
  bucket = aws_s3_bucket.lakehouse.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "lakehouse" {
  bucket                  = aws_s3_bucket.lakehouse.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket" "mlflow" {
  bucket_prefix = "${local.name}-mlflow-"
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "mlflow" {
  bucket                  = aws_s3_bucket.mlflow.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_glue_catalog_database" "lakehouse" {
  name = replace("${local.name}_lakehouse", "-", "_")
}

resource "aws_db_subnet_group" "main" {
  name       = local.name
  subnet_ids = aws_subnet.data[*].id
}

resource "random_password" "database" {
  length  = 32
  special = false
}

resource "aws_db_instance" "postgres" {
  identifier                   = local.name
  engine                       = "postgres"
  engine_version               = "17.5"
  instance_class               = "db.t4g.micro"
  allocated_storage            = 20
  max_allocated_storage        = 100
  storage_encrypted            = true
  db_name                      = "energy"
  username                     = "energy"
  password                     = random_password.database.result
  db_subnet_group_name         = aws_db_subnet_group.main.name
  vpc_security_group_ids       = [aws_security_group.runtime.id]
  publicly_accessible          = false
  backup_retention_period      = 7
  deletion_protection          = true
  skip_final_snapshot          = false
  final_snapshot_identifier    = "${local.name}-final"
  performance_insights_enabled = true
  apply_immediately            = false
}

resource "aws_secretsmanager_secret" "database_url" {
  name = "${local.name}/database-url"
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id = aws_secretsmanager_secret.database_url.id
  secret_string = jsonencode({
    DATABASE_URL = "postgresql+psycopg://energy:${urlencode(random_password.database.result)}@${aws_db_instance.postgres.address}:5432/energy"
  })
}

resource "aws_msk_serverless_cluster" "main" {
  cluster_name = local.name
  vpc_config {
    subnet_ids         = aws_subnet.data[*].id
    security_group_ids = [aws_security_group.runtime.id]
  }
  client_authentication {
    sasl {
      iam {
        enabled = true
      }
    }
  }
}

resource "aws_emrserverless_application" "spark" {
  name          = "${local.name}-spark"
  release_label = "emr-7.5.0"
  type          = "spark"
  auto_stop_configuration {
    enabled              = true
    idle_timeout_minutes = 15
  }
  maximum_capacity {
    cpu    = "8 vCPU"
    memory = "32 GB"
    disk   = "100 GB"
  }
}

resource "aws_ecr_repository" "images" {
  for_each             = toset(["api", "jobs", "mlflow"])
  name                 = "${local.name}/${each.key}"
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_efs_file_system" "models" {
  encrypted = true
  tags      = { Name = "${local.name}-models" }
}

resource "aws_efs_backup_policy" "models" {
  file_system_id = aws_efs_file_system.models.id
  backup_policy { status = "ENABLED" }
}

resource "aws_efs_access_point" "models" {
  file_system_id = aws_efs_file_system.models.id
  posix_user {
    uid = 10001
    gid = 10001
  }
  root_directory {
    path = "/models"
    creation_info {
      owner_uid   = 10001
      owner_gid   = 10001
      permissions = "0750"
    }
  }
}

resource "aws_efs_mount_target" "models" {
  count           = 2
  file_system_id  = aws_efs_file_system.models.id
  subnet_id       = aws_subnet.data[count.index].id
  security_groups = [aws_security_group.runtime.id]
}

resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 30
}

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_secrets" {
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = concat(
        [aws_secretsmanager_secret.database_url.arn],
        var.entsoe_token_secret_arn == "" ? [] : [var.entsoe_token_secret_arn]
      )
    }]
  })
}

resource "aws_iam_role" "ecs_task" {
  name = "${local.name}-ecs-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" }, Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_data" {
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], Resource = [aws_s3_bucket.lakehouse.arn, "${aws_s3_bucket.lakehouse.arn}/*", aws_s3_bucket.mlflow.arn, "${aws_s3_bucket.mlflow.arn}/*"] },
      { Effect = "Allow", Action = ["kafka-cluster:Connect", "kafka-cluster:DescribeCluster", "kafka-cluster:ReadData", "kafka-cluster:WriteData"], Resource = ["*"] }
    ]
  })
}

resource "aws_ecs_task_definition" "api" {
  count                    = local.runtime_enabled ? 1 : 0
  family                   = "${local.name}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 512
  memory                   = 1024
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn
  volume {
    name = "models"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.models.id
      transit_encryption = "ENABLED"
      root_directory     = "/"
      authorization_config {
        access_point_id = aws_efs_access_point.models.id
        iam             = "DISABLED"
      }
    }
  }
  container_definitions = jsonencode([{
    name         = "api", image = var.api_image, essential = true,
    portMappings = [{ containerPort = 8000, protocol = "tcp" }],
    secrets = concat(
      [{ name = "DATABASE_URL", valueFrom = "${aws_secretsmanager_secret.database_url.arn}:DATABASE_URL::" }],
      var.entsoe_token_secret_arn == "" ? [] : [{ name = "ENTSOE_TOKEN", valueFrom = var.entsoe_token_secret_arn }]
    ),
    environment = [
      { name = "APP_ENV", value = var.environment },
      { name = "MODEL_ARTIFACT_ROOT", value = "/models" },
      { name = "ICEBERG_WAREHOUSE", value = "s3://${aws_s3_bucket.lakehouse.id}/warehouse" },
      { name = "KAFKA_BOOTSTRAP_SERVERS", value = aws_msk_serverless_cluster.main.bootstrap_brokers_sasl_iam }
    ],
    mountPoints      = [{ sourceVolume = "models", containerPath = "/models", readOnly = false }],
    logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.api.name, awslogs-region = var.aws_region, awslogs-stream-prefix = "api" } },
    healthCheck      = { command = ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')\""], interval = 30, timeout = 5, retries = 3, startPeriod = 30 }
  }])
}

resource "aws_lb" "api" {
  count              = local.runtime_enabled ? 1 : 0
  name               = substr("${local.name}-api", 0, 32)
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "api" {
  count       = local.runtime_enabled ? 1 : 0
  name        = substr("${local.name}-api", 0, 32)
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"
  health_check {
    path    = "/health/ready"
    matcher = "200"
  }
}

resource "aws_lb_listener" "https" {
  count             = local.runtime_enabled ? 1 : 0
  load_balancer_arn = aws_lb.api[0].arn
  port              = 443
  protocol          = "HTTPS"
  certificate_arn   = var.certificate_arn
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api[0].arn
  }
}

resource "aws_ecs_service" "api" {
  count           = local.runtime_enabled ? 1 : 0
  name            = "${local.name}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api[0].arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.runtime.id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.api[0].arn
    container_name   = "api"
    container_port   = 8000
  }
  depends_on = [aws_lb_listener.https, aws_efs_mount_target.models]
}

resource "aws_iam_role" "scheduler" {
  name = "${local.name}-scheduler"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Principal = { Service = "scheduler.amazonaws.com" }, Action = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "scheduler" {
  role = aws_iam_role.scheduler.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [
      { Effect = "Allow", Action = ["ecs:RunTask"], Resource = local.runtime_enabled ? ["${local.task_definition_family_arn}:*"] : ["*"] },
      { Effect = "Allow", Action = ["iam:PassRole"], Resource = [aws_iam_role.ecs_execution.arn, aws_iam_role.ecs_task.arn] }
    ]
  })
}

resource "aws_scheduler_schedule_group" "forecasting" {
  name = "${local.name}-forecasting"
}

resource "aws_scheduler_schedule" "demand" {
  count               = local.runtime_enabled && var.entsoe_token_secret_arn != "" ? 1 : 0
  name                = "${local.name}-demand-quarter-hour"
  group_name          = aws_scheduler_schedule_group.forecasting.name
  schedule_expression = "rate(15 minutes)"
  flexible_time_window { mode = "OFF" }
  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn
    input = jsonencode({
      containerOverrides = [{ name = "api", command = ["energy-grid", "live-inference", "--target", "demand"] }]
    })
    ecs_parameters {
      task_definition_arn = local.task_definition_family_arn
      launch_type         = "FARGATE"
      network_configuration {
        subnets          = aws_subnet.public[*].id
        security_groups  = [aws_security_group.runtime.id]
        assign_public_ip = true
      }
    }
  }
}

resource "aws_scheduler_schedule" "price" {
  count                        = local.runtime_enabled && var.entsoe_token_secret_arn != "" ? 1 : 0
  name                         = "${local.name}-price-day-ahead"
  group_name                   = aws_scheduler_schedule_group.forecasting.name
  schedule_expression          = "cron(0 10 * * ? *)"
  schedule_expression_timezone = "Europe/Madrid"
  flexible_time_window { mode = "OFF" }
  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn
    input = jsonencode({
      containerOverrides = [{ name = "api", command = ["energy-grid", "live-inference", "--target", "price"] }]
    })
    ecs_parameters {
      task_definition_arn = local.task_definition_family_arn
      launch_type         = "FARGATE"
      network_configuration {
        subnets          = aws_subnet.public[*].id
        security_groups  = [aws_security_group.runtime.id]
        assign_public_ip = true
      }
    }
  }
}

resource "aws_scheduler_schedule" "candidate_training" {
  count                        = local.runtime_enabled && var.entsoe_token_secret_arn != "" ? 1 : 0
  name                         = "${local.name}-candidate-training"
  group_name                   = aws_scheduler_schedule_group.forecasting.name
  schedule_expression          = "cron(0 3 ? * SUN *)"
  schedule_expression_timezone = "Europe/Madrid"
  flexible_time_window { mode = "OFF" }
  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler.arn
    input = jsonencode({
      containerOverrides = [{
        name    = "api"
        command = ["energy-grid", "live-demo", "--no-publish-kafka", "--no-register-mlflow"]
      }]
    })
    ecs_parameters {
      task_definition_arn = local.task_definition_family_arn
      launch_type         = "FARGATE"
      network_configuration {
        subnets          = aws_subnet.public[*].id
        security_groups  = [aws_security_group.runtime.id]
        assign_public_ip = true
      }
    }
  }
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]
}

resource "aws_iam_role" "github_deploy" {
  name = "${local.name}-github-deploy"
  assume_role_policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect    = "Allow", Principal = { Federated = aws_iam_openid_connect_provider.github.arn }, Action = "sts:AssumeRoleWithWebIdentity",
      Condition = { StringEquals = { "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com" }, StringLike = { "token.actions.githubusercontent.com:sub" = "repo:${var.github_repository}:*" } }
    }]
  })
}

resource "aws_iam_role_policy" "github_deploy" {
  role = aws_iam_role.github_deploy.id
  policy = jsonencode({
    Version = "2012-10-17", Statement = [{
      Effect = "Allow", Action = ["ecr:GetAuthorizationToken", "ecr:BatchCheckLayerAvailability", "ecr:CompleteLayerUpload", "ecr:InitiateLayerUpload", "ecr:PutImage", "ecr:UploadLayerPart", "ecs:DescribeServices", "ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition", "ecs:UpdateService", "iam:PassRole"], Resource = "*"
    }]
  })
}

resource "aws_budgets_budget" "monthly" {
  name         = "${local.name}-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.monthly_budget_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  dynamic "notification" {
    for_each = var.alert_email == "" ? [] : [var.alert_email]
    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = 80
      threshold_type             = "PERCENTAGE"
      notification_type          = "FORECASTED"
      subscriber_email_addresses = [notification.value]
    }
  }
}
