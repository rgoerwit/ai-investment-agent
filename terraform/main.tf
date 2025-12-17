# Multi-Agent Trading System - Core Azure Infrastructure
# This module defines the reusable infrastructure components for all environments
# Updated to align with Azure provider 4.x patterns and current best practices
# UPDATED FOR GEMINI 3 MIGRATION (NOV 2025)

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # ═══════════════════════════════════════════════════════════════════════════
  # OPTIONAL: Remote Backend Configuration
  # ═══════════════════════════════════════════════════════════════════════════
  # Uncomment to store Terraform state in Azure Storage (recommended for teams)
  #
  # backend "azurerm" {
  #   resource_group_name  = "terraform-state-rg"
  #   storage_account_name = "tfstate<your-unique-id>"
  #   container_name       = "tfstate"
  #   key                  = "investment-agent.terraform.tfstate"
  # }
  #
  # Setup instructions:
  # 1. Create resource group: az group create -n terraform-state-rg -l eastus
  # 2. Create storage account: az storage account create -n tfstate<id> -g terraform-state-rg
  # 3. Create container: az storage container create -n tfstate --account-name tfstate<id>
  # 4. Uncomment backend block above
  # 5. Run: terraform init -migrate-state
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    log_analytics_workspace {
      permanently_delete_on_destroy = true
    }
    application_insights {
      disable_generated_rule = false
    }
  }
}

# Generate random suffix for globally unique names
resource "random_id" "main" {
  byte_length = 4
}

# Local values for consistent resource naming and tagging
locals {
  # Ensure names are DNS-compliant and globally unique
  name_suffix = lower(random_id.main.hex)

  # Common tags applied to all resources
  common_tags = merge(var.tags, {
    Environment = var.environment
    Project     = "MultiAgentTradingSystem"
    ManagedBy   = "Terraform"
    Repository  = "multi-agent-trading-system"
    CreatedDate = formatdate("YYYY-MM-DD", timestamp())
    Terraform   = "true"
  })

  # FIXED: Simplified storage account name generation with guaranteed compliance
  # Storage account names must be 3-24 characters, lowercase alphanumeric only
  # Format: st<env><random> where env is limited to 4 chars max
  env_short            = substr(lower(var.environment), 0, 4)
  storage_account_name = "st${local.env_short}${local.name_suffix}"

  # Validation: Storage name must be 3-24 chars and alphanumeric only
  storage_name_valid = (
    length(local.storage_account_name) >= 3 &&
    length(local.storage_account_name) <= 24 &&
    can(regex("^[a-z0-9]+$", local.storage_account_name))
  )

  # Final storage account name with fallback if validation fails
  storage_name_final = local.storage_name_valid ? local.storage_account_name : "st${substr(local.env_short, 0, 2)}${local.name_suffix}"

  # Container instance FQDN
  container_fqdn = "${lower(var.resource_group_name)}-${local.name_suffix}.${var.location}.azurecontainer.io"

  # Application name for consistent naming
  app_name = "trading-${var.environment}"
}

# Resource Group - Container for all environment resources
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = local.common_tags
}

# Log Analytics Workspace for centralized logging
resource "azurerm_log_analytics_workspace" "main" {
  name                       = "${var.resource_group_name}-logs"
  location                   = azurerm_resource_group.main.location
  resource_group_name        = azurerm_resource_group.main.name
  sku                        = "PerGB2018"
  retention_in_days          = var.log_retention_days
  daily_quota_gb             = var.log_daily_quota_gb
  internet_ingestion_enabled = true
  internet_query_enabled     = true

  tags = local.common_tags
}

# Application Insights for application performance monitoring
resource "azurerm_application_insights" "main" {
  name                = "${var.resource_group_name}-insights"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  workspace_id        = azurerm_log_analytics_workspace.main.id
  application_type    = "other"

  tags = local.common_tags
}

# Storage Account for agent memory persistence with enhanced security
resource "azurerm_storage_account" "main" {
  name                     = local.storage_name_final
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = var.storage_replication_type
  min_tls_version          = "TLS1_2"
  account_kind             = "StorageV2"
  access_tier              = "Hot"

  # Enhanced security settings
  https_traffic_only_enabled       = true
  public_network_access_enabled    = true
  allow_nested_items_to_be_public  = false
  cross_tenant_replication_enabled = false

  # Blob properties for better performance and security
  blob_properties {
    cors_rule {
      allowed_headers    = ["*"]
      allowed_methods    = ["GET", "HEAD", "POST", "PUT"]
      allowed_origins    = ["*"]
      exposed_headers    = ["*"]
      max_age_in_seconds = 3600
    }

    delete_retention_policy {
      days = 7
    }

    container_delete_retention_policy {
      days = 7
    }

    versioning_enabled = true
  }

  # Network access rules - Allow Azure services by default
  network_rules {
    default_action = "Allow" # For demo purposes; restrict in production
    bypass         = ["AzureServices"]

    # Fixed: Use ip_rules as a list directly, not as dynamic block
    ip_rules = var.allowed_ip_ranges
  }

  tags = local.common_tags

  # Add lifecycle rule to prevent accidental deletion of storage account
  lifecycle {
    prevent_destroy = false # Set to true in production
  }
}

# File Share for persistent agent memory storage
resource "azurerm_storage_share" "agent_memory" {
  name               = "agent-memory"
  storage_account_id = azurerm_storage_account.main.id
  quota              = var.storage_quota_gb
  access_tier        = "Hot"

  depends_on = [azurerm_storage_account.main]
}

# Additional file share for results output
resource "azurerm_storage_share" "results" {
  name               = "results"
  storage_account_id = azurerm_storage_account.main.id
  quota              = 10 # 10GB for results
  access_tier        = "Hot"

  depends_on = [azurerm_storage_account.main]
}

# Container Group for the trading system application
resource "azurerm_container_group" "main" {
  name                = "${var.resource_group_name}-aci"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  os_type             = "Linux"
  restart_policy      = var.restart_policy
  ip_address_type     = "Public"
  dns_name_label      = "${lower(var.resource_group_name)}-${local.name_suffix}"

  # Enhanced diagnostics configuration with conditional enablement
  dynamic "diagnostics" {
    for_each = var.enable_container_insights ? [1] : []
    content {
      log_analytics {
        workspace_id  = azurerm_log_analytics_workspace.main.id
        workspace_key = azurerm_log_analytics_workspace.main.primary_shared_key

        log_type = "ContainerInsights"
        metadata = {
          node-name   = "trading-agent"
          environment = var.environment
        }
      }
    }
  }

  # Main trading application container
  container {
    name   = "trading-agent"
    image  = var.docker_image
    cpu    = var.container_cpu
    memory = var.container_memory

    # ═══════════════════════════════════════════════════════════════════════
    # Health Probes - IMPORTANT NOTE FOR BATCH JOBS
    # ═══════════════════════════════════════════════════════════════════════
    # Azure Container Instances only support HTTP-based health checks.
    # This application is a batch job (no HTTP server by default).
    #
    # OPTIONS:
    # 1. Comment out health checks entirely (batch jobs don't need them)
    # 2. Add a simple HTTP health endpoint using Flask/FastAPI
    #
    # COMMENTED OUT: Health checks disabled for batch job
    # To enable, add HTTP endpoint to src/main.py and uncomment below:
    #
    # liveness_probe {
    #   http_get {
    #     path   = "/health"
    #     port   = 8080
    #     scheme = "Http"
    #   }
    #   initial_delay_seconds = 30
    #   period_seconds        = 60
    #   timeout_seconds       = 10
    #   failure_threshold     = 3
    # }
    #
    # readiness_probe {
    #   http_get {
    #     path   = "/ready"
    #     port   = 8080
    #     scheme = "Http"
    #   }
    #   initial_delay_seconds = 15
    #   period_seconds        = 30
    #   timeout_seconds       = 5
    #   success_threshold     = 1
    #   failure_threshold     = 3
    # }

    # Port configuration (only needed if adding HTTP endpoint)
    # ports {
    #   port     = 8080
    #   protocol = "TCP"
    # }

    # Application environment variables with enhanced configuration
    environment_variables = {
      # Core application settings
      "ENVIRONMENT"    = var.environment
      "DEFAULT_TICKER" = var.ticker_to_analyze
      "LOG_LEVEL"      = var.log_level

      # Model selection - configured via environment variables
      # See src/config.py for defaults if not set
      "LLM_PROVIDER" = "google"
      "QUICK_MODEL"  = var.quick_model
      "DEEP_MODEL"   = var.deep_model

      # LangSmith configuration
      "LANGSMITH_TRACING" = "true"
      "LANGSMITH_PROJECT" = "Deep-Trading-System-${var.environment}-Gemini"

      # Storage paths
      "CHROMA_PERSIST_DIR" = "/app/chroma_db"
      "RESULTS_DIR"        = "/app/results"

      # Azure configuration
      "AZURE_LOCATION" = var.location

      # Performance tuning
      "PYTHONUNBUFFERED"        = "1"
      "PYTHONDONTWRITEBYTECODE" = "1"

      # Container-specific settings
      "CONTAINER_MODE" = "true"
      "MAX_WORKERS"    = "1" # Single worker for container instance
    }

    # Secure environment variables for API keys
    # NOTE: For production, consider Azure Key Vault integration:
    # 1. Create Azure Key Vault: azurerm_key_vault
    # 2. Store secrets in Key Vault: azurerm_key_vault_secret
    # 3. Use managed identity to access: azurerm_user_assigned_identity
    # 4. Reference secrets via Key Vault CSI driver or init containers
    secure_environment_variables = {
      # UPDATED: Pass GOOGLE_API_KEY instead of OPENAI_API_KEY
      "GOOGLE_API_KEY"                        = var.google_api_key
      "FINNHUB_API_KEY"                       = var.finnhub_api_key
      "TAVILY_API_KEY"                        = var.tavily_api_key
      "LANGSMITH_API_KEY"                     = var.langsmith_api_key
      "APPLICATIONINSIGHTS_CONNECTION_STRING" = azurerm_application_insights.main.connection_string
    }

    # Volume mount for persistent memory storage
    volume {
      name                 = "agent-memory"
      mount_path           = "/app/chroma_db"
      read_only            = false
      share_name           = azurerm_storage_share.agent_memory.name
      storage_account_name = azurerm_storage_account.main.name
      storage_account_key  = azurerm_storage_account.main.primary_access_key
    }

    # Volume for results output
    volume {
      name                 = "results-volume"
      mount_path           = "/app/results"
      read_only            = false
      share_name           = azurerm_storage_share.results.name
      storage_account_name = azurerm_storage_account.main.name
      storage_account_key  = azurerm_storage_account.main.primary_access_key
    }

    # Resource limits and requests
    commands = [] # Use default container entrypoint
  }

  # Identity configuration for Azure service authentication
  dynamic "identity" {
    for_each = var.enable_managed_identity ? [1] : []
    content {
      type = "SystemAssigned"
    }
  }

  tags = local.common_tags
}

# Network Security Group for container access control
resource "azurerm_network_security_group" "main" {
  count               = length(var.allowed_ip_ranges) > 0 ? 1 : 0
  name                = "${var.resource_group_name}-nsg"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name

  # Allow HTTP health checks
  security_rule {
    name                       = "AllowHealthCheck"
    priority                   = 1001
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range     = "8080"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }

  # Dynamic rules for allowed IP ranges
  dynamic "security_rule" {
    for_each = { for idx, ip in var.allowed_ip_ranges : idx => ip }
    content {
      name                       = "AllowSpecificIP-${security_rule.key}"
      priority                   = 1100 + security_rule.key
      direction                  = "Inbound"
      access                     = "Allow"
      protocol                   = "Tcp"
      source_port_range          = "*"
      destination_port_range     = "8080"
      source_address_prefix      = security_rule.value
      destination_address_prefix = "*"
    }
  }

  tags = local.common_tags
}

# Data source for current client configuration
data "azurerm_client_config" "current" {}

# Role assignment for managed identity (if enabled)
resource "azurerm_role_assignment" "container_contributor" {
  count                = var.enable_managed_identity ? 1 : 0
  scope                = azurerm_resource_group.main.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_container_group.main.identity[0].principal_id

  depends_on = [azurerm_container_group.main]
}

# FIXED: Standardized Data Protection Backup Vault (modern approach)
resource "azurerm_data_protection_backup_vault" "main" {
  count               = var.enable_backup ? 1 : 0
  name                = "${var.resource_group_name}-backup-vault"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  datastore_type      = "VaultStore"
  redundancy          = "LocallyRedundant"

  # NOTE: For enhanced security with Azure Key Vault integration:
  # encryption {
  #   use_system_assigned_identity = true
  #   infrastructure_encryption_enabled = true
  # }

  tags = local.common_tags
}

# Optional: Monitor Action Group for alerts
resource "azurerm_monitor_action_group" "main" {
  count               = var.environment == "prod" ? 1 : 0
  name                = "${var.resource_group_name}-alerts"
  resource_group_name = azurerm_resource_group.main.name
  short_name          = "trading"

  # Email notifications (configure as needed)
  email_receiver {
    name          = "admin"
    email_address = "admin@example.com" # Replace with actual email
  }

  tags = local.common_tags
}

# Container Instance CPU alert
resource "azurerm_monitor_metric_alert" "high_cpu" {
  count               = var.environment == "prod" ? 1 : 0
  name                = "${var.resource_group_name}-high-cpu"
  resource_group_name = azurerm_resource_group.main.name
  scopes              = [azurerm_container_group.main.id]
  description         = "High CPU usage on trading container"

  criteria {
    metric_namespace = "Microsoft.ContainerInstance/containerGroups"
    metric_name      = "CpuUsage"
    aggregation      = "Average"
    operator         = "GreaterThan"
    threshold        = 80
  }

  action {
    action_group_id = azurerm_monitor_action_group.main[0].id
  }

  tags = local.common_tags
}