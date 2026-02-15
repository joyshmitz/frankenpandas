use serde::{Deserialize, Serialize};

use crate::asupersync::error::AsupersyncError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrityProof {
    pub algorithm: String,
    pub expected_digest: String,
    pub observed_digest: String,
    pub verified: bool,
}

pub trait IntegrityVerifier {
    fn verify(
        &self,
        artifact_id: &str,
        bytes: &[u8],
        expected_digest: &str,
    ) -> Result<IntegrityProof, AsupersyncError>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Fnv1aVerifier;

impl IntegrityVerifier for Fnv1aVerifier {
    fn verify(
        &self,
        artifact_id: &str,
        bytes: &[u8],
        expected_digest: &str,
    ) -> Result<IntegrityProof, AsupersyncError> {
        let observed_digest = fnv1a_hex(bytes);
        if observed_digest != expected_digest {
            return Err(AsupersyncError::IntegrityMismatch {
                artifact_id: artifact_id.to_string(),
                expected: expected_digest.to_string(),
                observed: observed_digest,
            });
        }

        Ok(IntegrityProof {
            algorithm: "fnv1a64".to_string(),
            expected_digest: expected_digest.to_string(),
            observed_digest,
            verified: true,
        })
    }
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}
