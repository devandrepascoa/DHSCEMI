# Requirements Document

## Introduction

This document specifies the requirements for a cost-aware autoscaling feature for an LLM inference proxy server. The feature enables intelligent hardware selection that minimizes cost-per-token by dynamically choosing between CPU configurations (1, 4, 8 cores) and GPU configurations (50%, 100%) based on workload and pricing. This is designed for a thesis on dynamic hardware selection for ML inference.

## Glossary

- **Cost_Aware_Autoscaler**: The component responsible for making scaling decisions based on cost-per-token optimization
- **Hardware_Config**: A specific hardware configuration with either CPU cores (1, 4, 8) or GPU percentage (50%, 100%)
- **Cost_Per_Token**: The calculated metric: hourly_cost / (tokens_per_second × 3600)

## Requirements

### Requirement 1: Hardware Configuration with Pricing and Throughput

**User Story:** As a system operator, I want each hardware configuration to have an associated hourly cost and model-specific throughput, so that the autoscaler can compute cost-per-token.

#### Acceptance Criteria

1. THE Hardware_Config SHALL store a $/hour cost
2. THE Hardware_Config SHALL have a unique identifier (config_id)
3. THE Cost_Aware_Autoscaler SHALL store pre-calculated tokens-per-second values per (model, config_id) pair
4. THE Hardware_Config SHALL support CPU configurations (1, 4, 8 cores)
5. THE Hardware_Config SHALL support GPU configurations (50%, 100% GPU)
6. THE Hardware_Config SHALL use the appropriate Docker image (full for CPU, full-cuda for GPU)
7. WHEN a model has no configured throughput for a config, THE Cost_Aware_Autoscaler SHALL use default throughput values

### Requirement 2: Cost-Per-Token Calculation

**User Story:** As a system operator, I want to compute the cost-per-token for each hardware configuration and model, so that I can compare cost efficiency.

#### Acceptance Criteria

1. THE Cost_Aware_Autoscaler SHALL calculate cost_per_token as: hourly_cost / (tokens_per_second × 3600)
2. THE Cost_Aware_Autoscaler SHALL compute cost_per_token using model-specific throughput values

### Requirement 3: Workload-Based Configuration Selection

**User Story:** As a system operator, I want the autoscaler to select the cheapest configuration that can handle the current workload, so that I minimize costs while meeting demand.

#### Acceptance Criteria

1. THE Cost_Aware_Autoscaler SHALL track tokens-per-second demand based on recent requests using a configurable sliding window (default 60s)
2. THE Cost_Aware_Autoscaler SHALL select the Hardware_Config with the lowest cost_per_token that can meet the current demand
3. WHEN multiple Hardware_Configs can meet demand, THE Cost_Aware_Autoscaler SHALL choose the one with lowest cost_per_token
4. THE Cost_Aware_Autoscaler SHALL consider both CPU and GPU configs when selecting the optimal configuration

### Requirement 4: Hysteresis Control

**User Story:** As a system operator, I want the autoscaler to avoid rapid oscillation between configurations, so that the system remains stable.

#### Acceptance Criteria

1. WHEN a scaling action occurs, THE Cost_Aware_Autoscaler SHALL enforce a configurable cooldown period (default 300s) before the next scaling action
2. THE Cost_Aware_Autoscaler SHALL evaluate scaling decisions on each incoming request, but only act if the cooldown has elapsed

### Requirement 5: Graceful Scaling

**User Story:** As a system operator, I want scaling transitions to not disrupt in-flight requests.

#### Acceptance Criteria

1. WHEN scaling to a new Hardware_Config, THE Cost_Aware_Autoscaler SHALL start the new container before stopping the old one
2. WHEN the new container is ready, THE Cost_Aware_Autoscaler SHALL drain active requests from the old container before stopping it
3. THE Cost_Aware_Autoscaler SHALL enforce a maximum drain timeout (default 60s) to prevent indefinite waiting on stuck requests
