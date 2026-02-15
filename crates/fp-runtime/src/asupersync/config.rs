use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CxCapability {
    Spawn,
    Time,
    Random,
    Io,
    Remote,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    pub spawn: bool,
    pub time: bool,
    pub random: bool,
    pub io: bool,
    pub remote: bool,
}

impl CapabilitySet {
    #[must_use]
    pub const fn none() -> Self {
        Self {
            spawn: false,
            time: false,
            random: false,
            io: false,
            remote: false,
        }
    }

    #[must_use]
    pub const fn all() -> Self {
        Self {
            spawn: true,
            time: true,
            random: true,
            io: true,
            remote: true,
        }
    }

    #[must_use]
    pub const fn for_capability(capability: CxCapability) -> Self {
        match capability {
            CxCapability::Spawn => Self {
                spawn: true,
                ..Self::none()
            },
            CxCapability::Time => Self {
                time: true,
                ..Self::none()
            },
            CxCapability::Random => Self {
                random: true,
                ..Self::none()
            },
            CxCapability::Io => Self {
                io: true,
                ..Self::none()
            },
            CxCapability::Remote => Self {
                remote: true,
                ..Self::none()
            },
        }
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            spawn: self.spawn || other.spawn,
            time: self.time || other.time,
            random: self.random || other.random,
            io: self.io || other.io,
            remote: self.remote || other.remote,
        }
    }

    #[must_use]
    pub const fn satisfies(self, required: Self) -> bool {
        (!required.spawn || self.spawn)
            && (!required.time || self.time)
            && (!required.random || self.random)
            && (!required.io || self.io)
            && (!required.remote || self.remote)
    }
}

impl Default for CapabilitySet {
    fn default() -> Self {
        Self::none()
    }
}

pub trait RequiresCapabilities {
    fn required_capabilities(&self) -> CapabilitySet;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsupersyncConfig {
    pub max_decode_attempts: u32,
    pub max_repair_symbols: u32,
    pub max_transfer_ms: u64,
    pub capabilities: CapabilitySet,
}

impl Default for AsupersyncConfig {
    fn default() -> Self {
        Self {
            max_decode_attempts: 3,
            max_repair_symbols: 32,
            max_transfer_ms: 5_000,
            capabilities: CapabilitySet::all(),
        }
    }
}

impl AsupersyncConfig {
    #[must_use]
    pub fn with_capabilities(mut self, capabilities: CapabilitySet) -> Self {
        self.capabilities = capabilities;
        self
    }
}
