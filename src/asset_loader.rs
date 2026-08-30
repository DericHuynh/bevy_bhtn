//! Bevy asset loader for `.htn` files.
//!
//! Loads the raw text of an `.htn` file and parses it into an [`HtnDomain`]
//! wrapped in an [`HtnAsset`]. This lets a whole AI domain be a hot-reloadable
//! asset via the standard Bevy asset pipeline (e.g. `AssetServer::load("combat.htn")`).
//! asset pipeline (e.g. `AssetServer::load("combat.htn")`).

use bevy_asset::{io::Reader, Asset, AssetLoader, AsyncReadExt, LoadContext};
use bevy_reflect::TypePath;

use crate::domain::HtnDomain;
use crate::dsl::parse_htn;

/// The parsed domain, reloadable from an `.htn` asset.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct HtnAsset {
    /// The parsed HTN domain.
    pub domain: HtnDomain,
}

/// The (empty) settings accepted by the `.htn` loader.
#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HtnAssetSettings {}

/// Error type for the `.htn` asset loader, wrapping [`HtnError`].
#[derive(Debug, thiserror::Error)]
pub enum HtnAssetError {
    /// Failure parsing the `.htn` text.
    #[error("Failed to parse HTN asset: {0}")]
    Parse(#[from] crate::error::HtnError),
    /// An I/O error reading the asset bytes.
    #[error("IO error loading HTN asset: {0}")]
    Io(#[from] std::io::Error),
}

/// Loads `.htn` text from the asset server into an [`HtnAsset`].
#[derive(TypePath)]
pub struct HtnAssetLoader;

impl AssetLoader for HtnAssetLoader {
    type Asset = HtnAsset;
    type Settings = HtnAssetSettings;
    type Error = HtnAssetError;

    async fn load(
        &self,
        reader: &mut dyn Reader,
        _settings: &Self::Settings,
        _load_context: &mut LoadContext<'_>,
    ) -> Result<Self::Asset, Self::Error> {
        let mut value = String::new();
        reader.read_to_string(&mut value).await?;
        let domain = parse_htn(&value)?;
        Ok(HtnAsset { domain })
    }

    fn extensions(&self) -> &[&str] {
        &["htn"]
    }
}
