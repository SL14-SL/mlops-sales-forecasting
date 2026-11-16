data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_vpc" "training" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${local.name_prefix}-training"
  }
}

resource "aws_subnet" "training" {
  vpc_id                  = aws_vpc.training.id
  cidr_block              = "10.42.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = false

  tags = {
    Name = "${local.name_prefix}-training"
  }
}

resource "aws_internet_gateway" "training" {
  vpc_id = aws_vpc.training.id

  tags = {
    Name = "${local.name_prefix}-training"
  }
}

resource "aws_route_table" "training" {
  vpc_id = aws_vpc.training.id

  tags = {
    Name = "${local.name_prefix}-training"
  }
}

resource "aws_route" "training_internet" {
  route_table_id         = aws_route_table.training.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.training.id
}

resource "aws_route_table_association" "training" {
  subnet_id      = aws_subnet.training.id
  route_table_id = aws_route_table.training.id
}

resource "aws_security_group" "training" {
  name        = "${local.name_prefix}-training"
  description = "Training task: no inbound access; outbound HTTPS only."
  vpc_id      = aws_vpc.training.id

  tags = {
    Name = "${local.name_prefix}-training"
  }
}

# Development training tasks access S3, ECR and CloudWatch via public endpoints.
# Destination IPs are unrestricted, but only outbound TCP/443 is allowed.
# No inbound rules exist. Reassess for confidential workloads or production.
# trivy:ignore:AVD-AWS-0104
resource "aws_vpc_security_group_egress_rule" "training_https" {
  security_group_id = aws_security_group.training.id
  description       = "HTTPS access to AWS APIs, container registry and storage."
  ip_protocol       = "tcp"
  from_port         = 443
  to_port           = 443
  cidr_ipv4         = "0.0.0.0/0"
}

resource "aws_default_security_group" "training" {
  vpc_id = aws_vpc.training.id

  tags = {
    Name = "${local.name_prefix}-unused-default"
  }
}