//! Persisted edge rendering and startup-only GL configuration. MSAA needs a new SDL window.
//! Text itself is drawn from a coverage atlas either way; the mode decides vector logos.
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TextRendering {
    Msaa,
    /// Default in this fork: measured on an LG C5, 4x MSAA leaves text edges only three grey
    /// levels, which the panel's 2x upscale turns into visible steps.
    #[default]
    Analytic,
}

impl TextRendering {
    pub const ALL: [Self; 2] = [Self::Msaa, Self::Analytic];

    pub fn index(self) -> usize {
        match self {
            Self::Msaa => 0,
            Self::Analytic => 1,
        }
    }

    pub fn label_key(self) -> &'static str {
        match self {
            Self::Msaa => "textRenderingMsaa",
            Self::Analytic => "textRenderingAnalytic",
        }
    }
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<TextRendering, D::Error> {
    Ok(
        match serde_json::Value::deserialize(deserializer)?.as_str() {
            Some("msaa") => TextRendering::Msaa,
            _ => TextRendering::Analytic,
        },
    )
}

#[derive(Debug, Serialize)]
pub struct Configuration {
    pub mode: TextRendering,
    #[serde(skip)]
    pub sample_counts: &'static [u8],
}

impl Configuration {
    /// Environment overrides retain their diagnostic role, ahead of the saved preference.
    pub fn new(preference: TextRendering, msaa: Option<&str>) -> Self {
        let mode = match msaa {
            Some("4") => TextRendering::Msaa,
            Some("0") => TextRendering::Analytic,
            _ => preference,
        };
        let sample_counts: &'static [u8] = match mode {
            TextRendering::Msaa => &[4, 2, 0],
            TextRendering::Analytic => &[0],
        };
        Self {
            mode,
            sample_counts,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_is_the_default_and_msaa_falls_back_to_fewer_samples() {
        assert_eq!(TextRendering::default(), TextRendering::Analytic);
        let config = Configuration::new(TextRendering::Msaa, None);
        assert_eq!(config.sample_counts, &[4, 2, 0]);
        let analytic = Configuration::new(TextRendering::Analytic, None);
        assert_eq!(analytic.sample_counts, &[0]);
    }

    #[test]
    fn environment_overrides_take_precedence_over_both_saved_modes() {
        for preference in TextRendering::ALL {
            let msaa = Configuration::new(preference, Some("4"));
            assert_eq!(msaa.sample_counts, &[4, 2, 0]);
            let analytic = Configuration::new(preference, Some("0"));
            assert_eq!(analytic.sample_counts, &[0]);
        }
        let invalid = Configuration::new(TextRendering::Analytic, Some("invalid"));
        assert_eq!(invalid.sample_counts, &[0]);
    }
}
