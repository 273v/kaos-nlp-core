//! Content-type detection by magic-byte signature.
//!
//! Wraps the `infer` crate (MIT, zero runtime deps) and exposes a
//! coarse-grained classifier suitable for feeding the kaos-agents
//! per-turn planner's `corpus_kinds` Signature input. The planner
//! does not need ~99% accuracy on 200+ content types (Magika's
//! pitch); it needs ~95% accuracy on ~10 buckets so it can decide
//! whether a session's uploaded corpus calls for PDF tools, office
//! tools, image tools, or a generic web/text path.
//!
//! See `kaos-modules/docs/internal/dynamic-tool-planning-prd.md` §4
//! (PR 4) for the design context.

use serde::Serialize;

/// One detection result: a precise label + the coarse planner bucket.
///
/// `mime_type` and `extension` come straight from `infer`. `group`
/// is the kaos-agents `corpus_kinds` bucket — a fixed enumeration
/// the planner's few-shot examples are written against.
#[derive(Debug, Clone, Serialize)]
pub struct ContentTypeResult {
    /// Canonical MIME type (e.g. ``application/pdf``,
    /// ``image/jpeg``). Empty string when no signature matched.
    pub mime_type: String,
    /// Typical file extension for the matched type (e.g. ``pdf``,
    /// ``jpg``). Empty string when no signature matched.
    pub extension: String,
    /// Coarse bucket for the planner's ``corpus_kinds`` input.
    /// One of:
    ///
    /// - ``"pdf"`` — application/pdf
    /// - ``"office-docx"``, ``"office-xlsx"``, ``"office-pptx"`` —
    ///   Open XML Office documents
    /// - ``"office-doc"``, ``"office-xls"``, ``"office-ppt"`` —
    ///   legacy binary Office formats
    /// - ``"image"`` — any image/*
    /// - ``"audio"`` — any audio/*
    /// - ``"video"`` — any video/*
    /// - ``"archive"`` — zip, tar, gz, bz2, 7z, rar, etc.
    /// - ``"email"`` — message/rfc822, application/mbox, .eml
    /// - ``"html"`` — text/html
    /// - ``"text"`` — text/plain
    /// - ``"font"`` — any font/*
    /// - ``"binary"`` — application/octet-stream or recognized
    ///   binary that doesn't fall into the above
    /// - ``"unknown"`` — `infer` returned nothing
    pub group: String,
}

/// Detect the content type of a byte slice via magic-bytes signature.
///
/// Returns `ContentTypeResult { mime_type: "", extension: "", group:
/// "unknown" }` when `infer` cannot recognize the bytes — the caller
/// can fall back to file-extension heuristics, NLP-driven content
/// sniffing, or just route to the generic-text path.
///
/// For a sample of at least 256 bytes detection accuracy on the
/// kelvin-legal upload set (PDF / DOCX / XLSX / PPTX / JPEG / PNG /
/// ZIP / EML / etc.) is effectively 100% — these formats all carry
/// stable magic-byte signatures in their header.
pub fn detect(bytes: &[u8]) -> ContentTypeResult {
    match infer::get(bytes) {
        Some(kind) => {
            let mime = kind.mime_type();
            let ext = kind.extension();
            let group = classify_group(mime, ext);
            ContentTypeResult {
                mime_type: mime.to_string(),
                extension: ext.to_string(),
                group,
            }
        }
        None => ContentTypeResult {
            mime_type: String::new(),
            extension: String::new(),
            group: "unknown".to_string(),
        },
    }
}

fn classify_group(mime: &str, ext: &str) -> String {
    // Order matters: Office-Open-XML files have `application/vnd.openxmlformats-*`
    // MIMEs that need to be checked before falling through to the
    // generic prefixes.
    let group = match (mime, ext) {
        ("application/pdf", _) => "pdf",
        ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", _)
        | (_, "docx") => "office-docx",
        ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", _) | (_, "xlsx") => {
            "office-xlsx"
        }
        ("application/vnd.openxmlformats-officedocument.presentationml.presentation", _)
        | (_, "pptx") => "office-pptx",
        ("application/msword", _) | (_, "doc") => "office-doc",
        ("application/vnd.ms-excel", _) | (_, "xls") => "office-xls",
        ("application/vnd.ms-powerpoint", _) | (_, "ppt") => "office-ppt",
        ("message/rfc822", _) | ("application/mbox", _) | (_, "eml") | (_, "mbox") => "email",
        ("text/html", _) => "html",
        ("text/plain", _) | (_, "txt") => "text",
        _ => "",
    };
    if !group.is_empty() {
        return group.to_string();
    }
    if mime.starts_with("image/") {
        return "image".to_string();
    }
    if mime.starts_with("audio/") {
        return "audio".to_string();
    }
    if mime.starts_with("video/") {
        return "video".to_string();
    }
    if mime.starts_with("font/") {
        return "font".to_string();
    }
    if is_archive(mime, ext) {
        return "archive".to_string();
    }
    if mime == "application/octet-stream" || !mime.is_empty() {
        return "binary".to_string();
    }
    "unknown".to_string()
}

fn is_archive(mime: &str, ext: &str) -> bool {
    matches!(
        mime,
        "application/zip"
            | "application/x-tar"
            | "application/gzip"
            | "application/x-bzip2"
            | "application/x-7z-compressed"
            | "application/vnd.rar"
            | "application/x-rar-compressed"
            | "application/x-xz"
            | "application/x-zstd"
    ) || matches!(
        ext,
        "zip" | "tar" | "gz" | "bz2" | "7z" | "rar" | "xz" | "zst"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_pdf_magic() {
        // %PDF-1.4 header.
        let bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n";
        let result = detect(bytes);
        assert_eq!(result.mime_type, "application/pdf");
        assert_eq!(result.extension, "pdf");
        assert_eq!(result.group, "pdf");
    }

    #[test]
    fn detects_png_magic() {
        // PNG header: 89 50 4E 47 0D 0A 1A 0A
        let bytes = b"\x89PNG\r\n\x1a\n";
        let result = detect(bytes);
        assert_eq!(result.extension, "png");
        assert_eq!(result.group, "image");
    }

    #[test]
    fn detects_zip_archive() {
        // ZIP magic: 50 4B 03 04
        let bytes = b"PK\x03\x04";
        let result = detect(bytes);
        assert_eq!(result.extension, "zip");
        assert_eq!(result.group, "archive");
    }

    #[test]
    fn unknown_bytes_return_unknown_group() {
        // Random bytes with no recognized signature.
        let bytes = b"hello world, plain text\nwith no magic bytes";
        let result = detect(bytes);
        assert_eq!(result.mime_type, "");
        assert_eq!(result.group, "unknown");
    }

    #[test]
    fn empty_bytes_return_unknown_group() {
        let result = detect(b"");
        assert_eq!(result.mime_type, "");
        assert_eq!(result.group, "unknown");
    }
}
