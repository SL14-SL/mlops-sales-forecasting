terraform {
  backend "gcs" {
    bucket  = "mlops-terraform-state-mlops-demand-forecasting"
    prefix  = "terraform/state"
  }
}
