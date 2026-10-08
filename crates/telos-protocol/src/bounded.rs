//! Bounded collections enforcing type-level maximum capacities to prevent memory `DoS`.

use crate::error::{ProtocolError, Result};
use crate::varint::{decode_varint, encode_varint, varint_size};
use std::fmt;
use std::ops::Deref;

/// A UTF-8 string guaranteed to never exceed `MAX` bytes.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundedString<const MAX: usize>(String);

impl<const MAX: usize> BoundedString<MAX> {
    /// Attempts to create a `BoundedString`, returning an error if byte length exceeds `MAX`.
    pub fn new(s: impl Into<String>) -> Result<Self> {
        let s = s.into();
        if s.len() > MAX {
            Err(ProtocolError::StringTooLong {
                actual: s.len(),
                limit: MAX,
            })
        } else {
            Ok(Self(s))
        }
    }

    /// Accesses the underlying string slice.
    #[inline]
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consumes the wrapper and returns the underlying `String`.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }

    /// Encodes this string into a buffer (`VarInt` length + UTF-8 bytes).
    #[inline]
    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.0.len() as u32, buf);
        buf.extend_from_slice(self.0.as_bytes());
    }

    /// Decodes a string from cursor with bounded length check.
    #[inline]
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let len = decode_varint(cursor)? as usize;
        if len > MAX {
            return Err(ProtocolError::StringTooLong {
                actual: len,
                limit: MAX,
            });
        }
        if cursor.len() < len {
            return Err(ProtocolError::UnexpectedEof);
        }
        let bytes = &cursor[..len];
        *cursor = &cursor[len..];

        let s = std::str::from_utf8(bytes)
            .map_err(|e| ProtocolError::InvalidUtf8(e.to_string()))?
            .to_string();

        Ok(Self(s))
    }

    /// Byte size when serialized on the wire.
    #[inline]
    #[must_use]
    pub fn encoded_size(&self) -> usize {
        varint_size(self.0.len() as u32) + self.0.len()
    }
}

impl<const MAX: usize> Deref for BoundedString<MAX> {
    type Target = str;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const MAX: usize> fmt::Debug for BoundedString<MAX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl<const MAX: usize> fmt::Display for BoundedString<MAX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

impl<const MAX: usize> Default for BoundedString<MAX> {
    fn default() -> Self {
        Self(String::new())
    }
}

/// A heap vector guaranteed to never exceed `MAX` elements.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct BoundedVec<T, const MAX: usize>(Vec<T>);

impl<T, const MAX: usize> BoundedVec<T, MAX> {
    /// Attempts to construct a `BoundedVec`, returning an error if length exceeds `MAX`.
    pub fn new(v: Vec<T>) -> Result<Self> {
        if v.len() > MAX {
            Err(ProtocolError::VecTooLong {
                actual: v.len(),
                limit: MAX,
            })
        } else {
            Ok(Self(v))
        }
    }

    /// Creates an empty `BoundedVec`.
    #[must_use]
    pub const fn empty() -> Self {
        Self(Vec::new())
    }
}

impl<T, const MAX: usize> Default for BoundedVec<T, MAX> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T, const MAX: usize> BoundedVec<T, MAX> {
    /// Returns a slice over elements.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    /// Consumes wrapper and returns underlying `Vec<T>`.
    #[inline]
    #[must_use]
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }

    /// Pushes an element into the vector if under capacity.
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.0.len() >= MAX {
            Err(ProtocolError::VecTooLong {
                actual: self.0.len() + 1,
                limit: MAX,
            })
        } else {
            self.0.push(item);
            Ok(())
        }
    }

    /// Encodes this vector using a provided element encoder.
    pub fn encode_with<F>(&self, buf: &mut Vec<u8>, mut encode_elem: F)
    where
        F: FnMut(&T, &mut Vec<u8>),
    {
        encode_varint(self.0.len() as u32, buf);
        for item in &self.0 {
            encode_elem(item, buf);
        }
    }

    /// Decodes a vector using a provided element decoder.
    pub fn decode_with<F>(cursor: &mut &[u8], mut decode_elem: F) -> Result<Self>
    where
        F: FnMut(&mut &[u8]) -> Result<T>,
    {
        let count = decode_varint(cursor)? as usize;
        if count > MAX {
            return Err(ProtocolError::VecTooLong {
                actual: count,
                limit: MAX,
            });
        }

        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(decode_elem(cursor)?);
        }

        Ok(Self(items))
    }
}

impl<const MAX: usize> BoundedVec<u8, MAX> {
    /// Encodes a raw byte buffer with length prefix.
    pub fn encode(&self, buf: &mut Vec<u8>) {
        encode_varint(self.0.len() as u32, buf);
        buf.extend_from_slice(&self.0);
    }

    /// Decodes a raw byte buffer from cursor.
    pub fn decode(cursor: &mut &[u8]) -> Result<Self> {
        let count = decode_varint(cursor)? as usize;
        if count > MAX {
            return Err(ProtocolError::VecTooLong {
                actual: count,
                limit: MAX,
            });
        }
        if cursor.len() < count {
            return Err(ProtocolError::UnexpectedEof);
        }
        let bytes = cursor[..count].to_vec();
        *cursor = &cursor[count..];
        Ok(Self(bytes))
    }
}

impl<T, const MAX: usize> Deref for BoundedVec<T, MAX> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: fmt::Debug, const MAX: usize> fmt::Debug for BoundedVec<T, MAX> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_string_limits() {
        let valid = BoundedString::<16>::new("hello world").unwrap();
        assert_eq!(valid.as_str(), "hello world");

        let mut buf = Vec::new();
        valid.encode(&mut buf);
        let mut cursor = &buf[..];
        let decoded = BoundedString::<16>::decode(&mut cursor).unwrap();
        assert_eq!(decoded.as_str(), "hello world");

        let invalid = BoundedString::<4>::new("too long string");
        assert!(matches!(invalid, Err(ProtocolError::StringTooLong { .. })));
    }
}
