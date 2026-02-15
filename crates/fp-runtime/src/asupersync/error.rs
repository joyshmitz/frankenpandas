use thiserror::Error;

use crate::asupersync::config::CapabilitySet;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AsupersyncError {
    #[error("invalid configuration: {0}")]
    Configuration(&'static str),
    #[error("capability requirement not satisfied; required={required:?} available={available:?}")]
    CapabilityDenied {
        required: CapabilitySet,
        available: CapabilitySet,
    },
    #[error("artifact not found: {0}")]
    ArtifactNotFound(String),
    #[error("integrity mismatch for {artifact_id}; expected={expected} observed={observed}")]
    IntegrityMismatch {
        artifact_id: String,
        expected: String,
        observed: String,
    },
    #[error("codec failure: {0}")]
    Codec(String),
    #[error("transport failure: {0}")]
    Transport(String),
    #[error("recovery exhausted for {artifact_id} after {attempts} attempts")]
    RecoveryExhausted { artifact_id: String, attempts: u32 },
}
