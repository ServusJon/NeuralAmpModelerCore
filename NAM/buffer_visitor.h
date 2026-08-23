#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  NAMHUNT-ZUSATZ, nicht Upstream (github.com/sdatkinson/NeuralAmpModelerCore).
//  Beim naechsten Upstream-Sync: diese Datei und alle `VisitBuffers`-Methoden
//  in NAM/ sind der komplette Eingriff.
//
//  Zweck: reine ADRESSAUSKUNFT ueber die grossen internen Puffer eines Modells,
//  damit der Host sie per mlock resident halten kann. Der macOS-Speicher-
//  Kompressor drueckt sonst die selten beruehrten Teile (vor allem die
//  Conv1D-Ringpuffer bei grosser Dilation) weg und der Audio-Thread zahlt beim
//  Zurueckholen Decompress-Faults — gemessen als Warte-Bloecke bis 2,84 ms
//  (NAMHunt: Docs/work/befunde/BEFUND_ARBEITSSATZ.md §5b).
//
//  Vertrag:
//   * VisitBuffers aendert NICHTS. Es meldet (Zeiger, Bytes) je Region.
//   * Nur auf dem Message-/Setup-Pfad rufen (Laden, Reset, Slim-Wechsel),
//     nie aus process(). Die gemeldeten Adressen gelten bis zum naechsten
//     SetMaxBufferSize/Reset/Rebuild — danach neu erfragen.
//   * Funktionszeiger statt std::function: kein Heap, keine Abhaengigkeit auf
//     Hostcode, in jeder Uebersetzungseinheit von NAM/ benutzbar.
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

namespace nam
{
/// \brief Callback fuer VisitBuffers: bekommt Startadresse und Groesse einer Region.
using BufferVisitor = void (*)(void* ctx, const void* data, std::size_t bytes);

/// \brief Eine Eigen-Matrix/-Vektor melden (zusammenhaengender Block, column-major).
template <typename Derived>
inline void VisitEigen(BufferVisitor visit, void* ctx, const Eigen::PlainObjectBase<Derived>& m)
{
  if (visit != nullptr && m.size() > 0)
    visit(ctx, m.data(), (std::size_t)m.size() * sizeof(typename Derived::Scalar));
}

/// \brief Einen std::vector melden (nur der Datenblock, nicht das Vektorobjekt).
template <typename T>
inline void VisitVector(BufferVisitor visit, void* ctx, const std::vector<T>& v)
{
  if (visit != nullptr && !v.empty())
    visit(ctx, v.data(), v.size() * sizeof(T));
}
} // namespace nam
