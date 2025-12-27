# Description: This file declares all the input variables for the 'prod' environment.
# These variables are passed down to the root infrastructure module. Their
# values are supplied by the 'terraform.tfvars' file in this directory and by
# TF_VAR_ environment variables in the CI/CD pipeline.
#
# UPDATED FOR GEMINI 3 MIGRATION

# --- Required API Keys & Secrets ---
variable "google_api_key" {
  description = "Google AI API Key for Gemini inference."
  type        = string
  sensitive   = true
}
variable "finnhub_api_key" {
  description = "Finnhub API Key for financial data."
  type        = string
  sensitive   = true
}
variable "tavily_api_key" {
  description = "Tavily API Key for web search capabilities."
  type        = string
  sensitive   = true
}
variable "langsmith_api_key" {
  description = "LangSmith API Key for tracing and observability."
  type        = string
  sensitive   = true
}

# --- Environment & Naming Configuration ---
variable "environment" {
  description = "The deployment environment (e.g., dev, staging, prod)."
  type        = string
}
variable "resource_group_name" {
  description = "The name of the Azure Resource Group."
  type        = string
}
variable "location" {
  description = "The Azure region where resources will be deployed."
  type        = string
}

# --- Application Container Configuration ---
variable "docker_image" {
  description = "The Docker image to deploy (e.g., 'myregistry/trading-system:latest')."
  type        = string
}
variable "ticker_to_analyze" {
  description = "The default stock ticker symbol for the container to analyze."
  type        = string
}
variable "container_cpu" {
  description = "CPU cores allocated to the container."
  type        = number
}
variable "container_memory" {
  description = "Memory (in GB) allocated to the container."
  type        = number
}
variable "restart_policy" {
  description = "Restart policy for the container group (Always, OnFailure, Never)."
  type        = string
}
variable "log_level" {
  description = "Log level for the application inside the container (e.g., INFO, DEBUG)."
  type        = string
}

# --- Storage Configuration ---
variable "storage_quota_gb" {
  description = "The size (in GB) of the file share for agent memory."
  type        = number
}

variable "storage_replication_type" {
  description = "Storage account replication type."
  type        = string
  default     = "LRS"
}

# --- Monitoring Configuration ---
variable "log_retention_days" {
  description = "Number of days to retain logs in Log Analytics workspace."
  type        = number
  default     = 30
}

variable "log_daily_quota_gb" {
  description = "Daily quota in GB for Log Analytics workspace (-1 for unlimited)."
  type        = number
  default     = -1
}

# --- Optional Configuration ---
variable "enable_container_insights" {
  description = "Enable Container Insights monitoring."
  type        = bool
  default     = true
}

variable "enable_managed_identity" {
  description = "Enable system-assigned managed identity for the container group."
  type        = bool
  default     = false
}

variable "enable_backup" {
  description = "Enable backup for storage account."
  type        = bool
  default     = false
}

variable "allowed_ip_ranges" {
  description = "List of IP ranges allowed to access the container (empty list allows all)."
  type        = list(string)
  default     = []
}

variable "tags" {
  description = "Additional tags to apply to all resources."
  type        = map(string)
  default     = {}
}
