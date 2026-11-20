resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "4096"

  execution_role_arn = aws_iam_role.api_execution.arn
  task_role_arn      = aws_iam_role.api_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      essential = true
      image     = "${aws_ecr_repository.api.repository_url}@${var.api_image_digest}"

      user             = "10001:10001"
      workingDirectory = "/app"

      mountPoints    = []
      systemControls = []
      volumesFrom    = []

      portMappings = [
        {
          containerPort = 8000
          hostPort      = 8000
          protocol      = "tcp"
        }
      ]

      linuxParameters = {
        capabilities = {
          add  = []
          drop = ["ALL"]
        }
      }

      environment = [
        {
          name  = "APP_ENV"
          value = var.environment
        },
        {
          name  = "CLOUD_PROVIDER"
          value = "aws"
        },
        {
          name  = "S3_BUCKET_NAME"
          value = aws_s3_bucket.training.id
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "MLFLOW_TRACKING_URI"
          value = local.mlflow_tracking_uri
        },
        {
          name  = "PREFECT_API_URL"
          value = var.prefect_api_url
        }
      ]

      secrets = [
        {
          name      = "API_KEY"
          valueFrom = aws_secretsmanager_secret.api_key.arn
        }
      ]

      healthCheck = {
        command = [
          "CMD",
          "python",
          "-c",
          "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/readyz', timeout=5)"
        ]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 120
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.api.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "api"
        }
      }
    }
  ])

  depends_on = [
    aws_iam_role_policy.api_execution,
    aws_iam_role_policy.api_task
  ]
}

resource "aws_ecs_service" "api" {
  count           = var.api_serving_enabled ? 1 : 0
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.training.id
  task_definition = aws_ecs_task_definition.api.arn

  launch_type      = "FARGATE"
  platform_version = "1.4.0"
  desired_count    = var.api_desired_count

  health_check_grace_period_seconds = 180
  wait_for_steady_state             = true

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets = [
      aws_subnet.training.id,
      aws_subnet.api_secondary.id
    ]

    security_groups  = [aws_security_group.api_task.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api[0].arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [
    aws_lb_listener.api,
    aws_iam_role_policy.api_execution,
    aws_iam_role_policy.api_task,
    aws_route.training_internet,
    aws_route_table_association.training,
    aws_route_table_association.api_secondary
  ]

  timeouts {
    create = "20m"
    update = "20m"
  }
}