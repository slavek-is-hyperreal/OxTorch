# HPC Kernel Report - OxTorch CPU Backend

Ten dokument śledzi stopień optymalizacji natywnej dla operacji CPU.
Zgodnie z `docs/binary_distribution.md`, celujemy w **Static Dispatch** (ZERO kosztu wyboru w locie).

## Symbole:
- ✅ **STATIC** - Implementacja skompilowana na stałe dla danej cechy (np. `-C target-feature=+avx2`).
- ⚡ **ASM-INLINED** - Użycie asemblera lub intrinsics z pełnym inliningiem.
- ❌ **FALLBACK** - Użycie generycznego kodu Rusta.

## Status Operacji (Binary):

### Op: `add`
| Precyzja | x86 Scalar | AVX | AVX2 | AVX-512 | NEON | Dispatch |
|---|---|---|---|---|---|---|
| BF16 | ❌ | ✅ | ❌ | ❌ | ✅ | Static |
| F16  | ❌ | ✅ | ❌ | ❌ | ✅ | Static |
| F32  | ❌ | ✅ | ✅ | ❌ | ✅ | Static |
| I8   | ❌ | ❌ | ✅ | ❌ | ✅ | Static |

### Op: `sub`
| Precyzja | x86 Scalar | SSE2 | AVX2 | AVX-512 | NEON | Dispatch |
|---|---|---|---|---|---|---|
| BF16 | ❌ | ❌ | ❌ | ❌ | ✅ | Static |
| I8   | ❌ | ✅ | ✅ | ❌ | ✅ | Static |

... (Wypełniane w trakcie migracji)
