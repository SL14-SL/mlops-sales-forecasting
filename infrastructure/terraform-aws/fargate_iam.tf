resource "aws_cloudwatch_log_group" "fargate_training" {
  name              = "/ecs/${local.name_prefix}-training"
  retention_in_days = 14
}

data "aws_iam_policy_document" "ecs_task_trust" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.aws_account_id]
    }

    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:aws:ecs:${var.aws_region}:${var.aws_account_id}:*"]
    }
  }
}

resource "aws_iam_role" "fargate_execution" {
  name               = "${local.name_prefix}-ecs-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

data "aws_iam_policy_document" "fargate_execution" {
  dynamic "statement" {
    for_each = length(compact([var.mlflow_token_secret_arn, var.prefect_key_secret_arn])) > 0 ? [1] : []
    content {
      sid       = "InjectLifecycleCredentials"
      actions   = ["secretsmanager:GetSecretValue"]
      resources = compact([var.mlflow_token_secret_arn, var.prefect_key_secret_arn])
    }
  }

  statement {
    sid       = "AuthenticateToEcr"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "PullTrainingImage"

    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer"
    ]

    resources = [aws_ecr_repository.training.arn]
  }

  statement {
    sid = "WriteContainerLogs"

    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]

    resources = ["${aws_cloudwatch_log_group.fargate_training.arn}:*"]
  }
}

resource "aws_iam_role_policy" "fargate_execution" {
  name   = "training-image-and-logs"
  role   = aws_iam_role.fargate_execution.id
  policy = data.aws_iam_policy_document.fargate_execution.json
}

resource "aws_iam_role" "fargate_task" {
  name               = "${local.name_prefix}-ecs-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_task_trust.json
}

data "aws_iam_policy_document" "fargate_task" {
  statement {
    sid       = "ListLifecycleArtifacts"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.training.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["data", "data/*", "models", "models/*", "artifacts", "artifacts/*", "mlflow", "mlflow/*"]
    }
  }

  statement {
    sid     = "ManageLifecycleArtifacts"
    actions = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = [
      "${aws_s3_bucket.training.arn}/data/*",
      "${aws_s3_bucket.training.arn}/models/*",
      "${aws_s3_bucket.training.arn}/artifacts/*",
      "${aws_s3_bucket.training.arn}/mlflow/*"
    ]
  }

  statement {
    sid       = "RefreshApiService"
    actions   = ["ecs:DescribeServices", "ecs:UpdateService"]
    resources = ["arn:aws:ecs:${var.aws_region}:${var.aws_account_id}:service/${aws_ecs_cluster.training.name}/${local.name_prefix}-api"]
  }

  statement {
    sid       = "ReadBucketLocation"
    actions   = ["s3:GetBucketLocation"]
    resources = [aws_s3_bucket.training.arn]
  }

  statement {
    sid = "ReadTrainingInputs"

    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.training.arn}/training/input/*"
    ]
  }

  statement {
    sid = "WriteTrainingOutputs"

    actions = [
      "s3:PutObject",
      "s3:AbortMultipartUpload"
    ]

    resources = [
      "${aws_s3_bucket.training.arn}/training/output/*"
    ]
  }
}

resource "aws_iam_role_policy" "fargate_task" {
  name   = "training-data-and-results"
  role   = aws_iam_role.fargate_task.id
  policy = data.aws_iam_policy_document.fargate_task.json
}