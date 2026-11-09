//! Network policy: VoiceBridge processes audio and text 100% locally. User-configurable
//! endpoints (ASR server, LLM translation server) must therefore point to this machine
//! or the local network — never to an Internet host.

use std::net::IpAddr;

/// Returns `Ok(())` when `url` targets loopback, a private LAN address or a `.local`
/// / `.localhost` host name.
pub fn ensure_local_endpoint(url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url.trim()).map_err(|e| format!("URL invalide « {url} » : {e}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!("Schéma non autorisé dans « {url} » (http/https uniquement)."));
    }
    let host = parsed
        .host()
        .ok_or_else(|| format!("URL sans hôte : « {url} »"))?;
    let local = match host {
        url::Host::Domain(name) => is_local_hostname(name),
        url::Host::Ipv4(ip) => is_local_ip(IpAddr::V4(ip)),
        url::Host::Ipv6(ip) => is_local_ip(IpAddr::V6(ip)),
    };
    if local {
        Ok(())
    } else {
        Err(format!(
            "« {url} » n'est pas une adresse locale. VoiceBridge fonctionne 100 % en local : \
             utilisez 127.0.0.1, localhost ou une adresse de votre réseau local."
        ))
    }
}

pub fn is_local_endpoint(url: &str) -> bool {
    ensure_local_endpoint(url).is_ok()
}

fn is_local_hostname(name: &str) -> bool {
    let n = name.trim_end_matches('.').to_ascii_lowercase();
    n == "localhost" || n.ends_with(".localhost") || n.ends_with(".local")
}

pub fn is_local_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_loopback() || v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_local_ip(IpAddr::V4(v4));
            }
            let seg0 = v6.segments()[0];
            v6.is_loopback()
                || (seg0 & 0xfe00) == 0xfc00 // unique local fc00::/7
                || (seg0 & 0xffc0) == 0xfe80 // link local fe80::/10
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_local_endpoints() {
        for u in [
            "http://127.0.0.1:8000/v1/audio/transcriptions",
            "http://localhost:1234/v1/chat/completions",
            "http://192.168.1.20:11434/v1/chat/completions",
            "http://10.0.0.5/x",
            "http://[::1]:8000/",
            "http://studio.local:8000/",
        ] {
            assert!(is_local_endpoint(u), "{u} should be accepted");
        }
    }

    #[test]
    fn rejects_internet_endpoints() {
        for u in [
            "https://api.openai.com/v1/audio/transcriptions",
            "https://translate.googleapis.com/translate_a/single",
            "http://8.8.8.8/",
            "http://localhost.evil.com/",
            "file:///etc/passwd",
            "not a url",
        ] {
            assert!(!is_local_endpoint(u), "{u} should be rejected");
        }
    }
}
