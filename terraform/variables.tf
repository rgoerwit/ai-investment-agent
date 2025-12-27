# Multi-Agent Trading System - Input Variables
# This file defines all input variables for the core Terraform module,
# enabling parameterized deployments across different environments.
#
# UPDATED FOR GEMINI 3 MIGRATION (NOV 2025)

# ===== REQUIRED API KEYS & SECRETS =====
# These variables are marked as sensitive and must be provided via environment variables.

variable "google_api_key" {
  description = "Google AI API key for Gemini inference and embeddings"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.google_api_key) > 0
    error_message = "Google API key must not be empty."
  }
}

variable "finnhub_api_key" {
  description = "Finnhub API key for financial market data"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.finnhub_api_key) > 0
    error_message = "Finnhub API key must not be empty."
  }
}

variable "tavily_api_key" {
  description = "Tavily API key for web search capabilities"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.tavily_api_key) > 0
    error_message = "Tavily API key must not be empty."
  }
}

variable "langsmith_api_key" {
  description = "LangSmith API key for agent tracing and observability"
  type        = string
  sensitive   = true

  validation {
    condition     = length(var.langsmith_api_key) > 0
    error_message = "LangSmith API key must not be empty."
  }
}

# ===== ENVIRONMENT & NAMING CONFIGURATION =====

variable "environment" {
  description = "The deployment environment (dev, staging, prod)"
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "resource_group_name" {
  description = "The name of the Azure Resource Group"
  type        = string

  validation {
    condition     = length(var.resource_group_name) >= 3 && length(var.resource_group_name) <= 63
    error_message = "Resource group name must be between 3 and 63 characters."
  }

  validation {
    condition     = can(regex("^[a-zA-Z0-9-]+$", var.resource_group_name))
    error_message = "Resource group name must contain only alphanumeric characters and hyphens."
  }
}

variable "location" {
  description = "The Azure region where resources will be deployed"
  type        = string
  default     = "eastus"

  validation {
    condition = contains([
      "eastus", "eastus2", "westus", "westus2", "westus3",
      "centralus", "northcentralus", "southcentralus",
      "westcentralus", "canadacentral", "canadaeast",
      "northeurope", "westeurope", "uksouth", "ukwest",
      "francecentral", "germanywestcentral", "norwayeast",
      "switzerlandnorth", "swedencentral", "japaneast",
      "japanwest", "koreacentral", "koreasouth",
      "southeastasia", "eastasia", "australiaeast",
      "australiasoutheast", "brazilsouth", "southafricanorth"
    ], var.location)
    error_message = "Location must be a valid Azure region."
  }
}

# ===== APPLICATION CONTAINER CONFIGURATION =====

variable "docker_image" {
  description = "The Docker image to deploy (e.g., 'myregistry/trading-system:v3.1.0')"
  type        = string

  validation {
    condition     = length(var.docker_image) > 0
    error_message = "Docker image must not be empty."
  }

  validation {
    condition     = can(regex("^[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$", var.docker_image)) || can(regex("^[a-zA-Z0-9._/-]+$", var.docker_image))
    error_message = "Docker image must be in a valid format (e.g., 'registry/image:tag' or 'image:tag')."
  }
}

variable "ticker_to_analyze" {
  description = "The default stock ticker symbol for analysis"
  type        = string
  default     = "AAPL"

  validation {
    condition     = can(regex("^[A-Z]{1,5}$", var.ticker_to_analyze))
    error_message = "Ticker symbol must be 1-5 uppercase letters."
  }
}

variable "container_cpu" {
  description = "CPU cores allocated to the container (minimum 0.5)"
  type        = number

  validation {
    condition     = var.container_cpu >= 0.1 && var.container_cpu <= 8.0
    error_message = "Container CPU must be between 0.1 and 8.0 cores."
  }
}

variable "container_memory" {
  description = "Memory (in GB) allocated to the container (minimum 0.5)"
  type        = number

  validation {
    condition     = var.container_memory >= 0.5 && var.container_memory <= 16.0
    error_message = "Container memory must be between 0.5 and 16.0 GB."
  }
}

variable "restart_policy" {
  description = "Restart policy for the container group"
  type        = string
  default     = "OnFailure"

  validation {
    condition     = contains(["Always", "OnFailure", "Never"], var.restart_policy)
    error_message = "Restart policy must be one of: Always, OnFailure, Never."
  }
}

variable "log_level" {
  description = "Log level for the application (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
  type        = string
  default     = "INFO"

  validation {
    condition     = contains(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], var.log_level)
    error_message = "Log level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL."
  }
}

# ===== MODEL CONFIGURATION =====

variable "quick_model" {
  description = "Gemini model for data gathering agents (thinking_level=low)"
  type        = string
  default     = "gemini-3-pro-preview"
}

variable "deep_model" {
  description = "Gemini model for synthesis agents (thinking_level=high in normal mode)"
  type        = string
  default     = "gemini-3-pro-preview"
}

# ===== STORAGE CONFIGURATION =====

variable "storage_quota_gb" {
  description = "The size (in GB) of the file share for agent memory"
  type        = number
  default     = 10

  validation {
    condition     = var.storage_quota_gb >= 1 && var.storage_quota_gb <= 1000
    error_message = "Storage quota must be between 1 and 1000 GB."
  }
}

variable "storage_replication_type" {
  description = "Storage account replication type"
  type        = string
  default     = "LRS"

  validation {
    condition     = contains(["LRS", "GRS", "RAGRS", "ZRS", "GZRS", "RAGZRS"], var.storage_replication_type)
    error_message = "Storage replication type must be one of: LRS, GRS, RAGRS, ZRS, GZRS, RAGZRS."
  }
}

# ===== MONITORING CONFIGURATION =====

variable "log_retention_days" {
  description = "Number of days to retain logs in Log Analytics workspace"
  type        = number
  default     = 30

  validation {
    condition     = var.log_retention_days >= 7 && var.log_retention_days <= 730
    error_message = "Log retention must be between 7 and 730 days."
  }
}

variable "log_daily_quota_gb" {
  description = "Daily quota in GB for Log Analytics workspace (-1 for unlimited)"
  type        = number
  default     = -1

  validation {
    condition     = var.log_daily_quota_gb == -1 || (var.log_daily_quota_gb >= 1 && var.log_daily_quota_gb <= 1000)
    error_message = "Daily quota must be -1 (unlimited) or between 1 and 1000 GB."
  }
}

# ===== NETWORK CONFIGURATION =====

variable "allowed_ip_ranges" {
  description = "List of IP ranges allowed to access the container (empty list allows all)"
  type        = list(string)
  default     = []

  validation {
    condition = alltrue([
      for ip in var.allowed_ip_ranges : can(cidrhost(ip, 0))
    ])
    error_message = "All IP ranges must be valid CIDR blocks (e.g., '192.168.1.0/24')."
  }
}

# ===== OPTIONAL CONFIGURATION =====

variable "enable_container_insights" {
  description = "Enable Container Insights monitoring"
  type        = bool
  default     = true
}

variable "enable_managed_identity" {
  description = "Enable system-assigned managed identity for the container group"
  type        = bool
  default     = false
}

variable "enable_backup" {
  description = "Enable backup for storage account"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags to apply to all resources"
  type        = map(string)
  default     = {}

  validation {
    condition = alltrue([
      for k, v in var.tags : can(regex("^[a-zA-Z0-9_.-]+$", k)) && length(k) <= 512
    ])
    error_message = "Tag keys must be alphanumeric with underscores, hyphens, or periods, and max 512 characters."
  }

  validation {
    condition = alltrue([
      for k, v in var.tags : length(v) <= 256
    ])
    error_message = "Tag values must be 256 characters or less."
  }
}
