//! Update check and installation from GitHub Releases.
//!
//! Only public release metadata and the installer are downloaded: no audio, text,
//! configuration or identifier ever leaves the machine. The installer is verified
//! against the release's `SHA256SUMS.txt` before it is launched.

use futures_util::StreamExt;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const REPO: &str = "cclngit/VoiceBridge";
pub const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A newer release that can be installed on this platform.
#[derive(Debug, Clone)]
pub struct UpdateInfo {
    pub version: String,
    pub page_url: String,
    pub asset_name: String,
    pub asset_url: String,
    pub sha256: String,
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
    html_url: String,
    #[serde(default)]
    draft: bool,
    #[serde(default)]
    prerelease: bool,
    assets: Vec<Asset>,
}

#[derive(Deserialize)]
struct Asset {
    name: String,
    browser_download_url: String,
}

fn http_client(timeout: Duration) -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(concat!("VoiceBridge/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(10))
        .timeout(timeout)
        .build()
        .map_err(|e| format!("Client HTTP : {e}"))
}

/// `"v1.2.3"` / `"1.2.3"` → `(1, 2, 3)`.
pub fn parse_version(v: &str) -> Option<(u64, u64, u64)> {
    let core = v.trim().trim_start_matches(['v', 'V']);
    let core = core.split(['-', '+']).next()?;
    let mut it = core.split('.').map(|p| p.parse::<u64>().ok());
    let v = (it.next()??, it.next()??, it.next()??);
    it.next().is_none().then_some(v)
}

pub fn is_newer(candidate: &str, current: &str) -> bool {
    matches!((parse_version(candidate), parse_version(current)), (Some(a), Some(b)) if a > b)
}

/// Name prefix and extension of this platform's installer in a release.
fn platform_asset() -> Option<(&'static str, &'static str)> {
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        Some(("VoiceBridge-windows-x86_64-setup", ".exe"))
    } else if cfg!(all(target_os = "windows", target_arch = "aarch64")) {
        Some(("VoiceBridge-windows-arm64-setup", ".exe"))
    } else if cfg!(target_os = "macos") {
        Some(("VoiceBridge-macos-universal", ".dmg"))
    } else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        Some(("VoiceBridge-linux-amd64", ".deb"))
    } else if cfg!(all(target_os = "linux", target_arch = "aarch64")) {
        Some(("VoiceBridge-linux-arm64", ".deb"))
    } else {
        None
    }
}

/// Returns `Some(update)` when the latest GitHub release is newer than this build.
pub async fn check_latest() -> Result<Option<UpdateInfo>, String> {
    let client = http_client(Duration::from_secs(20))?;
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    let resp = client
        .get(&url)
        .header(reqwest::header::ACCEPT, "application/vnd.github+json")
        .send()
        .await
        .map_err(|e| format!("Serveur de mises à jour injoignable : {e}"))?;
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None); // no release published yet
    }
    let release: Release = resp
        .error_for_status()
        .map_err(|e| format!("Vérification des mises à jour : {e}"))?
        .json()
        .await
        .map_err(|e| format!("Réponse de mise à jour illisible : {e}"))?;

    if release.draft || release.prerelease || !is_newer(&release.tag_name, CURRENT_VERSION) {
        return Ok(None);
    }
    let version = release.tag_name.trim_start_matches(['v', 'V']).to_string();

    let (prefix, ext) = platform_asset().ok_or("Plateforme sans installeur publié.")?;
    let asset = release
        .assets
        .iter()
        .find(|a| a.name.starts_with(prefix) && a.name.ends_with(ext))
        .ok_or_else(|| format!("La version {version} ne contient pas d'installeur pour cette plateforme."))?;
    let sums = release
        .assets
        .iter()
        .find(|a| a.name == "SHA256SUMS.txt")
        .ok_or("La release ne publie pas SHA256SUMS.txt : mise à jour refusée.")?;

    let sums_text = client
        .get(&sums.browser_download_url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("SHA256SUMS.txt : {e}"))?
        .text()
        .await
        .map_err(|e| format!("SHA256SUMS.txt : {e}"))?;
    let sha256 = parse_sha256sums(&sums_text, &asset.name)
        .ok_or_else(|| format!("Empreinte SHA-256 absente pour {}.", asset.name))?;

    Ok(Some(UpdateInfo {
        version,
        page_url: release.html_url,
        asset_name: asset.name.clone(),
        asset_url: asset.browser_download_url.clone(),
        sha256,
    }))
}

/// Finds `name`'s hash in `sha256sum` output (`<hex>  <name>` or `<hex> *<name>`).
fn parse_sha256sums(text: &str, name: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (hash, file) = line.trim().split_once(char::is_whitespace)?;
        let file = file.trim().trim_start_matches('*');
        (file == name && hash.len() == 64 && hash.chars().all(|c| c.is_ascii_hexdigit()))
            .then(|| hash.to_ascii_lowercase())
    })
}

/// Downloads the installer into a temporary folder, verifies its SHA-256 and returns its path.
/// `progress` receives 0–100.
pub async fn download(info: &UpdateInfo, progress: Arc<AtomicU8>) -> Result<PathBuf, String> {
    // The asset name is used as a file name: refuse anything that could leave the folder.
    if info.asset_name.contains(['/', '\\']) || info.asset_name.contains("..") {
        return Err("Nom d'installeur invalide.".into());
    }
    let dir = std::env::temp_dir().join("VoiceBridge-update");
    tokio::fs::create_dir_all(&dir).await.map_err(|e| format!("Dossier temporaire : {e}"))?;
    let path = dir.join(&info.asset_name);

    let client = http_client(Duration::from_secs(30 * 60))?;
    let resp = client
        .get(&info.asset_url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("Téléchargement de la mise à jour : {e}"))?;
    let total = resp.content_length().unwrap_or(0);

    let mut file = tokio::fs::File::create(&path).await.map_err(|e| format!("Écriture : {e}"))?;
    let mut hasher = Sha256::new();
    let mut done: u64 = 0;
    let mut stream = resp.bytes_stream();
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Téléchargement interrompu : {e}"))?;
        hasher.update(&chunk);
        file.write_all(&chunk).await.map_err(|e| format!("Écriture : {e}"))?;
        done += chunk.len() as u64;
        if total > 0 {
            progress.store(((done * 100) / total).min(100) as u8, Ordering::Relaxed);
        }
    }
    file.flush().await.map_err(|e| format!("Écriture : {e}"))?;
    drop(file);

    let actual = format!("{:x}", hasher.finalize());
    if actual != info.sha256 {
        let _ = tokio::fs::remove_file(&path).await;
        return Err("Empreinte SHA-256 incorrecte : installeur supprimé, mise à jour annulée.".into());
    }
    progress.store(100, Ordering::Relaxed);
    Ok(path)
}

/// Starts the verified installer. On Windows it runs silently and the caller must quit
/// the app right after so the files can be replaced; the installer relaunches VoiceBridge.
/// On macOS / Linux the package is opened with the system installer.
pub fn launch_installer(path: &std::path::Path) -> Result<(), String> {
    let result = if cfg!(target_os = "windows") {
        std::process::Command::new(path)
            .args(["/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS"])
            .spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(path).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(path).spawn()
    };
    result.map(|_| ()).map_err(|e| format!("Impossible de lancer l'installeur : {e}"))
}

/// Whether installing quits and relaunches the app by itself (Windows) or hands over
/// to the system package installer (macOS / Linux).
pub const fn installs_silently() -> bool {
    cfg!(target_os = "windows")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_compare_numerically() {
        assert!(is_newer("v0.3.10", "0.3.9"));
        assert!(is_newer("1.0.0", "0.99.99"));
        assert!(!is_newer("v0.3.0", "0.3.0"));
        assert!(!is_newer("v0.2.9", "0.3.0"));
        assert!(!is_newer("garbage", "0.3.0"));
        assert_eq!(parse_version("v1.2.3-beta"), Some((1, 2, 3)));
        assert_eq!(parse_version("1.2"), None);
    }

    #[test]
    fn sha256sums_lines_are_matched_exactly() {
        let h = "a".repeat(64);
        let text = format!("{h}  VoiceBridge-windows-x86_64-setup-v0.3.1.exe\n{}  other.exe\n", "b".repeat(64));
        assert_eq!(parse_sha256sums(&text, "VoiceBridge-windows-x86_64-setup-v0.3.1.exe"), Some(h.clone()));
        assert_eq!(parse_sha256sums(&format!("{h} *x.dmg"), "x.dmg"), Some(h));
        assert_eq!(parse_sha256sums(&text, "missing.exe"), None);
    }
}
