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

      entryPoint = [
        "python",
        "-m",
        "mlops_sales_forecasting.aws.fargate_train"
      ]

      command = [
        "--config",
        "/app/configs/dev.yaml",
        "--input-uri",
        "s3://${aws_s3_bucket.training.id}/training/input/rossmann-v1/",
        "--output-uri",
        "s3://${aws_s3_bucket.training.id}/training/output",
        "--estimators",
        "20"
      ]

      environment = [
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
      ]

      linuxParameters = {
        capabilities = {
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