//! Minimal pure-Rust SentencePiece *decoder*.
//!
//! Kyutai models (STT and Hibiki) emit SentencePiece token ids. Decoding only needs the
//! piece vocabulary stored in the `.model` protobuf, so instead of linking the C++
//! `sentencepiece` library we parse the `ModelProto` directly:
//!
//! ```text
//! message ModelProto   { repeated SentencePiece pieces = 1; ... }
//! message SentencePiece { string piece = 1; float score = 2; Type type = 3; }
//! enum Type { NORMAL = 1; UNKNOWN = 2; CONTROL = 3; USER_DEFINED = 4; UNUSED = 5; BYTE = 6; }
//! ```

use std::path::Path;

const WORD_BOUNDARY: char = '\u{2581}'; // "▁"

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte(u8),
}

#[derive(Debug, Clone)]
struct Piece {
    text: String,
    kind: PieceType,
}

#[derive(Debug, Clone)]
pub struct SentencePieceDecoder {
    pieces: Vec<Piece>,
}

impl SentencePieceDecoder {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = std::fs::read(path.as_ref())
            .map_err(|e| format!("cannot read tokenizer {}: {e}", path.as_ref().display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut pieces = Vec::new();
        let mut r = Reader::new(bytes);
        while !r.is_empty() {
            let (field, wire) = r.key()?;
            if field == 1 && wire == 2 {
                let msg = r.bytes()?;
                pieces.push(parse_piece(msg)?);
            } else {
                r.skip(wire)?;
            }
        }
        if pieces.is_empty() {
            return Err("tokenizer model contains no pieces".into());
        }
        Ok(Self { pieces })
    }

    #[cfg(test)]
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Decode a single piece, keeping its leading space marker (useful for streaming).
    pub fn decode_piece(&self, id: u32) -> String {
        let mut bytes = Vec::new();
        self.push_piece(id, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a sequence of ids into text (leading space removed, like SentencePiece).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            self.push_piece(id, &mut bytes);
        }
        let s = String::from_utf8_lossy(&bytes).into_owned();
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }

    fn push_piece(&self, id: u32, out: &mut Vec<u8>) {
        let Some(piece) = self.pieces.get(id as usize) else {
            return;
        };
        match piece.kind {
            PieceType::Control | PieceType::Unused => {}
            PieceType::Unknown => out.extend_from_slice(" \u{2047} ".as_bytes()),
            PieceType::Byte(b) => out.push(b),
            PieceType::Normal | PieceType::UserDefined => {
                let text = piece.text.replace(WORD_BOUNDARY, " ");
                out.extend_from_slice(text.as_bytes());
            }
        }
    }
}

fn parse_piece(msg: &[u8]) -> Result<Piece, String> {
    let mut r = Reader::new(msg);
    let mut text = String::new();
    let mut kind_id = 1u64;
    while !r.is_empty() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => text = String::from_utf8_lossy(r.bytes()?).into_owned(),
            (3, 0) => kind_id = r.varint()?,
            _ => r.skip(wire)?,
        }
    }
    let kind = match kind_id {
        2 => PieceType::Unknown,
        3 => PieceType::Control,
        4 => PieceType::UserDefined,
        5 => PieceType::Unused,
        6 => {
            // Byte pieces are spelled "<0xAB>".
            let hex = text.trim_start_matches("<0x").trim_end_matches('>');
            PieceType::Byte(u8::from_str_radix(hex, 16).unwrap_or(b'?'))
        }
        _ => PieceType::Normal,
    };
    Ok(Piece { text, kind })
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, String> {
        let mut result = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or("truncated varint")?;
            self.pos += 1;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err("varint too long".into())
    }

    fn key(&mut self) -> Result<(u64, u8), String> {
        let k = self.varint()?;
        Ok((k >> 3, (k & 7) as u8))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        let s = self.buf.get(self.pos..end).ok_or("truncated field")?;
        self.pos = end;
        Ok(s)
    }

    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let len = self.varint()? as usize;
        self.take(len)
    }

    fn skip(&mut self, wire: u8) -> Result<(), String> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.take(8)?;
            }
            2 => {
                self.bytes()?;
            }
            5 => {
                self.take(4)?;
            }
            w => return Err(format!("unsupported protobuf wire type {w}")),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn varint(mut v: u64, out: &mut Vec<u8>) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                break;
            }
            out.push(b | 0x80);
        }
    }

    fn piece(text: &str, kind: u64) -> Vec<u8> {
        let mut p = vec![0x0a];
        varint(text.len() as u64, &mut p);
        p.extend_from_slice(text.as_bytes());
        p.extend_from_slice(&[0x15, 0, 0, 0, 0]); // score (fixed32)
        p.push(0x18);
        varint(kind, &mut p);
        let mut field = vec![0x0a];
        varint(p.len() as u64, &mut field);
        field.extend(p);
        field
    }

    fn model() -> SentencePieceDecoder {
        let mut bytes = Vec::new();
        for (t, k) in [
            ("<unk>", 2),
            ("<s>", 3),
            ("</s>", 3),
            ("▁Bonjour", 1),
            ("▁le", 1),
            ("▁monde", 1),
            ("!", 1),
            ("<0xC3>", 6),
            ("<0xA9>", 6),
        ] {
            bytes.extend(piece(t, k));
        }
        // An unrelated trailer field (trainer spec, field 2) must be skipped.
        bytes.extend_from_slice(&[0x12, 0x02, 0x08, 0x01]);
        SentencePieceDecoder::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn decodes_words_and_skips_control_tokens() {
        let m = model();
        assert_eq!(m.vocab_size(), 9);
        assert_eq!(m.decode(&[1, 3, 4, 5, 6, 2]), "Bonjour le monde!");
        assert_eq!(m.decode_piece(4), " le");
    }

    #[test]
    fn decodes_byte_fallback_pieces() {
        let m = model();
        assert_eq!(m.decode(&[7, 8]), "é");
    }
}
