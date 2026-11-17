locals {
  mlflow_tracking_uri     = var.mlflow_managed_enabled ? aws_apigatewayv2_api.mlflow[0].api_endpoint : var.mlflow_tracking_uri
  mlflow_token_secret_arn = var.mlflow_managed_enabled ? aws_secretsmanager_secret.mlflow_token[0].arn : var.mlflow_token_secret_arn
}

resource "aws_ecr_repository" "mlflow" {
  count                = var.mlflow_managed_enabled ? 1 : 0
  name                 = "${local.name_prefix}-mlflow"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false
  encryption_configuration { encryption_type = "AES256" }
  image_scanning_configuration { scan_on_push = true }
}

resource "aws_secretsmanager_secret" "mlflow_token" {
  count                   = var.mlflow_managed_enabled ? 1 : 0
  name                    = "${local.name_prefix}/mlflow-token"
  recovery_window_in_days = 7
}

resource "aws_cloudwatch_log_group" "mlflow" {
  count             = var.mlflow_managed_enabled ? 1 : 0
  name              = "/ecs/${local.name_prefix}-mlflow"
  retention_in_days = 14
}

resource "aws_security_group" "mlflow" {
  count       = var.mlflow_managed_enabled ? 1 : 0
  name        = "${local.name_prefix}-mlflow"
  description = "MLflow: ingress only from the internal load balancer."
  vpc_id      = aws_vpc.training.id
}

resource "aws_security_group" "mlflow_db" {
  count       = var.mlflow_managed_enabled ? 1 : 0
  name        = "${local.name_prefix}-mlflow-db"
  description = "PostgreSQL: ingress only from MLflow."
  vpc_id      = aws_vpc.training.id
}

resource "aws_vpc_security_group_ingress_rule" "mlflow_db" {
  count                        = var.mlflow_managed_enabled ? 1 : 0
  security_group_id            = aws_security_group.mlflow_db[0].id
  referenced_security_group_id = aws_security_group.mlflow[0].id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
}

resource "aws_vpc_security_group_egress_rule" "mlflow_db" {
  count                        = var.mlflow_managed_enabled ? 1 : 0
  security_group_id            = aws_security_group.mlflow[0].id
  referenced_security_group_id = aws_security_group.mlflow_db[0].id
  ip_protocol                  = "tcp"
  from_port                    = 5432
  to_port                      = 5432
}

# AWS APIs are accessed through public HTTPS endpoints; no NAT gateway is required.
# trivy:ignore:AVD-AWS-0104
resource "aws_vpc_security_group_egress_rule" "mlflow_https" {
  count             = var.mlflow_managed_enabled ? 1 : 0
  security_group_id = aws_security_group.mlflow[0].id
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_db_subnet_group" "mlflow" {
  count      = var.mlflow_managed_enabled ? 1 : 0
  name       = "${local.name_prefix}-mlflow"
  subnet_ids = [aws_subnet.training.id, aws_subnet.api_secondary.id]
}

resource "aws_db_instance" "mlflow" {
  count                       = var.mlflow_managed_enabled ? 1 : 0
  identifier                  = "${local.name_prefix}-mlflow"
  engine                      = "postgres"
  engine_version              = "16"
  instance_class              = "db.t4g.micro"
  allocated_storage           = 20
  max_allocated_storage       = 30
  storage_type                = "gp3"
  storage_encrypted           = true
  db_name                     = "mlflow"
  username                    = "mlflow_admin"
  manage_master_user_password = true
  publicly_accessible         = false
  multi_az                    = false
  db_subnet_group_name        = aws_db_subnet_group.mlflow[0].name
  vpc_security_group_ids      = [aws_security_group.mlflow_db[0].id]
  backup_retention_period     = 7
  auto_minor_version_upgrade  = true
  copy_tags_to_snapshot       = true
  deletion_protection         = var.mlflow_database_deletion_protection
  skip_final_snapshot         = false
  final_snapshot_identifier   = var.mlflow_final_snapshot_identifier != "" ? var.mlflow_final_snapshot_identifier : "${local.name_prefix}-mlflow-final"
  apply_immediately           = true
}

resource "aws_iam_role" "mlflow_execution" {
  count              = var.mlflow_managed_enabled ? 1 : 0
  name               = "${local.name_prefix}-mlflow-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

resource "aws_iam_role_policy" "mlflow_execution" {
  count = var.mlflow_managed_enabled ? 1 : 0
  role  = aws_iam_role.mlflow_execution[0].id
  name  = "image-and-logs"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["ecr:GetAuthorizationToken"], Resource = "*" },
      { Effect = "Allow", Action = ["ecr:BatchCheckLayerAvailability", "ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"], Resource = aws_ecr_repository.mlflow[0].arn },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.mlflow[0].arn}:*" }
    ]
  })
}

resource "aws_iam_role" "mlflow_task" {
  count              = var.mlflow_managed_enabled ? 1 : 0
  name               = "${local.name_prefix}-mlflow-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

resource "aws_iam_role_policy" "mlflow_task" {
  count = var.mlflow_managed_enabled ? 1 : 0
  role  = aws_iam_role.mlflow_task[0].id
  name  = "read-rds-credentials-and-artifacts"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = aws_db_instance.mlflow[0].master_user_secret[0].secret_arn },
      { Effect = "Allow", Action = ["s3:GetObject"], Resource = "${aws_s3_bucket.training.arn}/mlflow/*" },
      { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.training.arn, Condition = { StringLike = { "s3:prefix" = ["mlflow", "mlflow/*"] } } }
    ]
  })
}

resource "aws_ecs_task_definition" "mlflow" {
  count                    = var.mlflow_managed_enabled ? 1 : 0
  family                   = "${local.name_prefix}-mlflow"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.mlflow_execution[0].arn
  task_role_arn            = aws_iam_role.mlflow_task[0].arn
  container_definitions = jsonencode([{
    name            = "mlflow"
    essential       = true
    image           = "${aws_ecr_repository.mlflow[0].repository_url}@${var.mlflow_image_digest}"
    user            = "10001:10001"
    portMappings    = [{ containerPort = 5000, hostPort = 5000, protocol = "tcp" }]
    linuxParameters = { capabilities = { drop = ["ALL"] } }
    environment = [
      { name = "AWS_REGION", value = var.aws_region },
      { name = "AWS_DEFAULT_REGION", value = var.aws_region },
      { name = "MLFLOW_DB_HOST", value = aws_db_instance.mlflow[0].address },
      { name = "MLFLOW_DB_SECRET_ARN", value = aws_db_instance.mlflow[0].master_user_secret[0].secret_arn },
      { name = "MLFLOW_S3_ARTIFACT_ROOT", value = "s3://${aws_s3_bucket.training.id}/mlflow" },
      { name = "MLFLOW_ALLOWED_HOSTS", value = "${replace(aws_apigatewayv2_api.mlflow[0].api_endpoint, "https://", "")},${aws_lb.mlflow[0].dns_name},127.0.0.1:5000,localhost:5000,10.42.*:5000" },
      { name = "MLFLOW_DISABLE_TELEMETRY", value = "true" }
    ]
    healthCheck = {
      command  = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5)"]
      interval = 30, timeout = 10, retries = 3, startPeriod = 180
    }
    logConfiguration = { logDriver = "awslogs", options = {
      awslogs-group = aws_cloudwatch_log_group.mlflow[0].name, awslogs-region = var.aws_region, awslogs-stream-prefix = "mlflow"
    } }
  }])
  lifecycle {
    precondition {
      condition     = can(regex("^sha256:[a-f0-9]{64}$", var.mlflow_image_digest))
      error_message = "Build the MLflow image and provide its SHA-256 digest."
    }
  }
  depends_on = [aws_iam_role_policy.mlflow_execution, aws_iam_role_policy.mlflow_task]
}

resource "aws_ecs_service" "mlflow" {
  count                              = var.mlflow_managed_enabled ? 1 : 0
  name                               = "${local.name_prefix}-mlflow"
  cluster                            = aws_ecs_cluster.training.id
  task_definition                    = aws_ecs_task_definition.mlflow[0].arn
  launch_type                        = "FARGATE"
  platform_version                   = "1.4.0"
  desired_count                      = var.mlflow_desired_count
  wait_for_steady_state              = true
  health_check_grace_period_seconds  = 240
  deployment_maximum_percent         = 100
  deployment_minimum_healthy_percent = 0
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  network_configuration {
    subnets          = [aws_subnet.training.id, aws_subnet.api_secondary.id]
    security_groups  = [aws_security_group.mlflow[0].id]
    assign_public_ip = true
  }
  load_balancer {
    target_group_arn = aws_lb_target_group.mlflow[0].arn
    container_name   = "mlflow"
    container_port   = 5000
  }
  depends_on = [aws_lb_listener.mlflow, aws_route.training_internet, aws_route_table_association.training, aws_route_table_association.api_secondary]
  timeouts {
    create = "25m"
    update = "25m"
  }
}
