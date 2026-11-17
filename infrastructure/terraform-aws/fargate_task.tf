resource "aws_ecs_cluster" "training" {
  name = "${local.name_prefix}-training"
}

resource "aws_ecs_task_definition" "training" {
  family                   = "${local.name_prefix}-training"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "2048"
  memory                   = "16384"

  execution_role_arn = aws_iam_role.fargate_execution.arn
  task_role_arn      = aws_iam_role.fargate_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "training"
      essential = true
      image = join("@", [
        aws_ecr_repository.training.repository_url,
        var.fargate_training_image_digest
      ])

      user             = "10001:10001"
      workingDirectory = "/tmp"

      mountPoints    = []
      portMappings   = []
      systemControls = []
      volumesFrom    = []

      entryPoint = [
        "python",
        "-m",
        "mlops_sales_forecasting.aws.lifecycle"
      ]

      command = ["train"]

      secrets = concat(
        var.mlflow_token_secret_arn != "" ? [{
          name      = "MLFLOW_TRACKING_TOKEN"
          valueFrom = var.mlflow_token_secret_arn
        }] : [],
        var.prefect_key_secret_arn != "" ? [{
          name      = "PREFECT_API_KEY"
          valueFrom = var.prefect_key_secret_arn
        }] : []
      )

      environment = concat([
        { name = "CLOUD_PROVIDER", value = "aws" },
        { name = "APP_ENV", value = var.environment },
        { name = "S3_BUCKET_NAME", value = aws_s3_bucket.training.id },
        { name = "MLFLOW_TRACKING_URI", value = var.mlflow_tracking_uri },
        { name = "PREFECT_API_URL", value = var.prefect_api_url },
        {
          name  = "AWS_DEFAULT_REGION"
          value = var.aws_region
        },
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "HOME"
          value = "/tmp"
        }
        ], var.api_serving_enabled ? [
        { name = "AWS_ECS_CLUSTER", value = aws_ecs_cluster.training.arn },
        { name = "AWS_API_SERVICE", value = "${local.name_prefix}-api" },
        { name = "AWS_API_ENDPOINT", value = aws_apigatewayv2_api.api[0].api_endpoint }
      ] : [])

      linuxParameters = {
        capabilities = {
          add  = []
          drop = ["ALL"]
        }
      }

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.fargate_training.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "training"
        }
      }
    }
  ])

  depends_on = [
    aws_iam_role_policy.fargate_execution,
    aws_iam_role_policy.fargate_task
  ]
}