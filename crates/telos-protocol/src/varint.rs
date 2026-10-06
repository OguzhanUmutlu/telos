//! Variable-width integer codecs (LEB128) and `ZigZag` transformations.

use crate::error::{ProtocolError, Result};

/// Encodes an unsigned 32-bit integer as an LEB128 `VarInt`.
#[inline]
pub fn encode_varint(mut val: u32, buf: &mut Vec<u8>) {
    while val >= 0x80 {
        buf.push(((val & 0x7f) as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

/// Decodes an unsigned 32-bit integer from an LEB128 `VarInt`.
#[inline]
pub fn decode_varint(cursor: &mut &[u8]) -> Result<u32> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;

    for _ in 0..5 {
        if cursor.is_empty() {
            return Err(ProtocolError::UnexpectedEof);
        }
        let byte = cursor[0];
        *cursor = &cursor[1..];

        result |= u32::from(byte & 0x7f) << shift;
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }

    Err(ProtocolError::VarIntOverflow)
}

/// Computes the serialized byte length of an LEB128 `VarInt`.
#[inline]
#[must_use]
pub const fn varint_size(val: u32) -> usize {
    if val < (1 << 7) {
        1
    } else if val < (1 << 14) {
        2
    } else if val < (1 << 21) {
        3
    } else if val < (1 << 28) {
        4
    } else {
        5
    }
}

/// Encodes an unsigned 64-bit integer as an LEB128 `VarLong`.
#[inline]
pub fn encode_varlong(mut val: u64, buf: &mut Vec<u8>) {
    while val >= 0x80 {
        buf.push(((val & 0x7f) as u8) | 0x80);
        val >>= 7;
    }
    buf.push(val as u8);
}

/// Decodes an unsigned 64-bit integer from an LEB128 `VarLong`.
#[inline]
pub fn decode_varlong(cursor: &mut &[u8]) -> Result<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;

    for _ in 0..10 {
        if cursor.is_empty() {
            return Err(ProtocolError::UnexpectedEof);
        }
        let byte = cursor[0];
        *cursor = &cursor[1..];

        result |= u64::from(byte & 0x7f) << shift;
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
    }

    Err(ProtocolError::VarLongOverflow)
}

/// Computes the serialized byte length of an LEB128 `VarLong`.
#[inline]
#[must_use]
pub const fn varlong_size(val: u64) -> usize {
    if val < (1 << 7) {
        1
    } else if val < (1 << 14) {
        2
    } else if val < (1 << 21) {
        3
    } else if val < (1 << 28) {
        4
    } else if val < (1 << 35) {
        5
    } else if val < (1 << 42) {
        6
    } else if val < (1 << 49) {
        7
    } else if val < (1 << 56) {
        8
    } else if val < (1 << 63) {
        9
    } else {
        10
    }
}

/// Maps signed 32-bit integer to unsigned space using `ZigZag` encoding.
#[inline]
#[must_use]
pub const fn zigzag_i32(n: i32) -> u32 {
    ((n << 1) ^ (n >> 31)) as u32
}

/// Unmaps `ZigZag`-encoded 32-bit integer back to signed value.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub const fn unzigzag_i32(n: u32) -> i32 {
    ((n >> 1) as i32) ^ (-((n & 1) as i32))
}

/// Maps signed 64-bit integer to unsigned space using `ZigZag` encoding.
#[inline]
#[must_use]
pub const fn zigzag_i64(n: i64) -> u64 {
    ((n << 1) ^ (n >> 63)) as u64
}

/// Unmaps `ZigZag`-encoded 64-bit integer back to signed value.
#[inline]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub const fn unzigzag_i64(n: u64) -> i64 {
    ((n >> 1) as i64) ^ (-((n & 1) as i64))
}

/// Encodes signed 32-bit integer using `ZigZag` and LEB128 `VarInt`.
#[inline]
pub fn encode_varint_zigzag(n: i32, buf: &mut Vec<u8>) {
    encode_varint(zigzag_i32(n), buf);
}

/// Decodes signed 32-bit integer using LEB128 `VarInt` and `ZigZag`.
#[inline]
pub fn decode_varint_zigzag(cursor: &mut &[u8]) -> Result<i32> {
    let unsigned = decode_varint(cursor)?;
    Ok(unzigzag_i32(unsigned))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_round_trip() {
        let cases = [0, 1, 127, 128, 255, 16383, 16384, u32::MAX / 2, u32::MAX];
        for &val in &cases {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf);
            let mut cursor = &buf[..];
            let decoded = decode_varint(&mut cursor).unwrap();
            assert_eq!(val, decoded);
            assert!(cursor.is_empty());
        }
    }

    #[test]
    fn test_zigzag_round_trip() {
        let cases = [0, -1, 1, -128, 127, i32::MIN, i32::MAX];
        for &val in &cases {
            let encoded = zigzag_i32(val);
            let decoded = unzigzag_i32(encoded);
            assert_eq!(val, decoded);
        }
    }
}
