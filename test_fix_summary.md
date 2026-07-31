# GGUF Test Fixes Summary

## Problem
Tests in `dispatch_integration.rs` were failing with `InvalidValueType` errors when parsing synthetic GGUF v3 files.

## Root Cause
The test helper was writing array elements incorrectly:
1. **Array metadata order**: Was writing `count, element_type` instead of `element_type, count`
2. **String lengths in arrays**: Was using u64 (8 bytes) instead of u32 (4 bytes) for string lengths inside arrays

## Fixes Applied

### 1. Array Metadata Order (`pesti-runner/tests/dispatch_integration.rs`)
Changed from:
```rust
buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());  // count first
buf.extend_from_slice(&(element_type as u32).to_le_bytes());  // then element_type
```

To:
```rust
buf.extend_from_slice(&(element_type as u32).to_le_bytes());  // element_type first
buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());  // then count
```

### 2. String Lengths in Arrays (`pesti-runner/tests/dispatch_integration.rs`)
Changed from:
```rust
buf.extend_from_slice(&(s.len() as u64).to_le_bytes());  // 8 bytes
```

To:
```rust
buf.extend_from_slice(&(s.len() as u32).to_le_bytes());  // 4 bytes
```

### 3. Parser Update (`pesti-gguf/src/parser.rs`)
Changed from:
```rust
let str_len = reader.read_u64::<LittleEndian>()? as usize;  // Read u64
```

To:
```rust
let str_len = reader.read_u32::<LittleEndian>()? as usize;  // Read u32
```

## Verification
After these fixes, the test GGUF file structure is:
- Magic: "GGUF" (4 bytes)
- Version: 3 (u32, 4 bytes)
- Tensor count: 12 (u64, 8 bytes)
- KV count: 13 (u64, 8 bytes)
- KV pairs: 13 entries with u64 key lengths
- Array structure: element_type(u32=10) + count(u64=10) + 10 strings
- Each string: length(u32=4) + data("tok0", 4 bytes)
- Tensor info: name_length(u64) + name + shape_count(u32) + ...

## Next Steps
1. Remove debug print statements
2. Test with real GGUF models (Option 3 from user's request)
3. Update test candidates or download a model
