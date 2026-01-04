-- AGI-in-a-Box Database Initialization
-- PostgreSQL schema for persistent state management

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";  -- For embedding storage (pgvector)

-- Model Profiles Table
-- Stores learned topological profiles for models
CREATE TABLE IF NOT EXISTS model_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) UNIQUE NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    tier VARCHAR(50) NOT NULL,
    persistence_centroid JSONB,  -- Learned persistence diagram centroid
    betti_0_mean FLOAT,
    betti_1_mean FLOAT,
    betti_2_mean FLOAT,
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    avg_latency_ms FLOAT,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_model_profiles_model_id ON model_profiles(model_id);
CREATE INDEX idx_model_profiles_tier ON model_profiles(tier);

-- Routing Decisions Log
-- Audit trail for routing decisions
CREATE TABLE IF NOT EXISTS routing_decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id UUID NOT NULL,
    query_hash VARCHAR(64),
    primary_model_id VARCHAR(255) NOT NULL,
    fallback_model_ids JSONB,
    confidence FLOAT NOT NULL,
    topology_complexity VARCHAR(50),
    betti_0 INTEGER,
    betti_1 INTEGER,
    betti_2 INTEGER,
    constraints_applied JSONB,
    diagnostics JSONB,
    execution_time_ms FLOAT,
    success BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_routing_decisions_request_id ON routing_decisions(request_id);
CREATE INDEX idx_routing_decisions_model_id ON routing_decisions(primary_model_id);
CREATE INDEX idx_routing_decisions_created_at ON routing_decisions(created_at);

-- Agent Memory Table
-- Persistent memory for CrewAI agents
CREATE TABLE IF NOT EXISTS agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,  -- short_term, long_term, entity
    content JSONB NOT NULL,
    embedding vector(1536),  -- Optional embedding for semantic search
    relevance_score FLOAT,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_agent_memory_agent_id ON agent_memory(agent_id);
CREATE INDEX idx_agent_memory_type ON agent_memory(memory_type);
CREATE INDEX idx_agent_memory_expires ON agent_memory(expires_at);

-- Workflow Executions
-- Track agent workflow runs
CREATE TABLE IF NOT EXISTS workflow_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- pending, running, completed, failed
    input_params JSONB,
    output_result JSONB,
    agents_involved JSONB,
    tasks_completed INTEGER DEFAULT 0,
    tasks_total INTEGER,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT
);

CREATE INDEX idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX idx_workflow_executions_name ON workflow_executions(workflow_name);

-- Model Metrics (for online learning)
CREATE TABLE IF NOT EXISTS model_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id VARCHAR(255) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    tags JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX idx_model_metrics_name ON model_metrics(metric_name);
CREATE INDEX idx_model_metrics_recorded ON model_metrics(recorded_at);

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_model_profiles_updated_at
    BEFORE UPDATE ON model_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_memory_updated_at
    BEFORE UPDATE ON agent_memory
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE OR REPLACE VIEW v_model_performance AS
SELECT
    model_id,
    model_name,
    tier,
    success_count,
    failure_count,
    CASE
        WHEN (success_count + failure_count) > 0
        THEN success_count::FLOAT / (success_count + failure_count)
        ELSE 0
    END as success_rate,
    avg_latency_ms,
    last_updated
FROM model_profiles;

CREATE OR REPLACE VIEW v_routing_stats AS
SELECT
    DATE_TRUNC('hour', created_at) as hour,
    primary_model_id,
    topology_complexity,
    COUNT(*) as request_count,
    AVG(confidence) as avg_confidence,
    AVG(execution_time_ms) as avg_execution_time,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as success_rate
FROM routing_decisions
GROUP BY DATE_TRUNC('hour', created_at), primary_model_id, topology_complexity;

-- Grant permissions (adjust for your security requirements)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO agi_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO agi_app;
