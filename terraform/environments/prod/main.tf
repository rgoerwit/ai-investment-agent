# Multi-Agent Trading System - Production Environment Configuration
# This file defines the Terraform backend and calls the main infrastructure
# module with variables specific to the 'prod' environment.
#
# UPDATED FOR GEMINI 3 MIGRATION

terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }

  backend "azurerm" {
    resource_group_name  = "tfstate-rg"
    storage_account_name = "tradingsystemtfstate" # This should be globally unique and created once
    container_name       = "tfstate"
    key                  = "prod.terraform.tfstate"
  }
}

provider "azurerm" {
  features {}
}

# Call the root module with prod-specific variables
module "trading_system" {
  source = "../../" # Go up two directories to the root module

  # Pass variables to the module
  environment         = var.environment
  resource_group_name = var.resource_group_name
  location            = var.location
  docker_image        = var.docker_image
  container_cpu       = var.container_cpu
  container_memory    = var.container_memory
  restart_policy      = var.restart_policy
  storage_quota_gb    = var.storage_quota_gb
  log_level           = var.log_level
  ticker_to_analyze   = var.ticker_to_analyze

  # Optional settings
  storage_replication_type  = var.storage_replication_type
  log_retention_days        = var.log_retention_days
  log_daily_quota_gb        = var.log_daily_quota_gb
  enable_container_insights = var.enable_container_insights
  enable_managed_identity   = var.enable_managed_identity
  enable_backup             = var.enable_backup
  allowed_ip_ranges         = var.allowed_ip_ranges
  tags                      = var.tags

  # Sensitive variables are passed through from the environment
  # UPDATED: Passing google_api_key instead of openai_api_key
  google_api_key    = var.google_api_key
  finnhub_api_key   = var.finnhub_api_key
  tavily_api_key    = var.tavily_api_key
  langsmith_api_key = var.langsmith_api_key
}
