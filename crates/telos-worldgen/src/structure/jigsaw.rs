//! Template jigsaw piece placement and spatial collision assembler.

use super::bounding_box::StructureBoundingBox;

/// Cardinal direction vector for jigsaw joint connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JigsawDirection {
    /// Negative Z
    North,
    /// Positive Z
    South,
    /// Positive X
    East,
    /// Negative X
    West,
    /// Positive Y
    Up,
    /// Negative Y
    Down,
}

impl JigsawDirection {
    /// Returns the unit delta `[dx, dy, dz]` for this direction.
    #[must_use]
    pub const fn to_offset(self) -> [i32; 3] {
        match self {
            Self::North => [0, 0, -1],
            Self::South => [0, 0, 1],
            Self::East => [1, 0, 0],
            Self::West => [-1, 0, 0],
            Self::Up => [0, 1, 0],
            Self::Down => [0, -1, 0],
        }
    }

    /// Returns the opposite direction.
    #[must_use]
    pub const fn opposite(self) -> Self {
        match self {
            Self::North => Self::South,
            Self::South => Self::North,
            Self::East => Self::West,
            Self::West => Self::East,
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }
}

/// A connection joint on a jigsaw structure piece.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JigsawJoint {
    /// Local offset `[x, y, z]` within the piece bounding box.
    pub local_pos: [i32; 3],
    /// Direction the joint faces outwards.
    pub direction: JigsawDirection,
    /// Connection type identifier (e.g. `"corridor"`, `"arch"`, `"wall"`).
    pub joint_type: &'static str,
}

/// Archetype kind for modular structure pieces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    /// Subterranean dungeon central monster chamber.
    DungeonChamber,
    /// Surface ruin perimeter wall segment.
    RuinWall,
    /// Surface ruin stone pillar / column.
    RuinPillar,
    /// Surface ruin crumbling archway.
    RuinArch,
    /// Surface ruin altar / central courtyard.
    RuinAltar,
    /// Surface ruin sunken vault / basement room.
    RuinVault,
}

/// An instantiated structure piece with a world-space bounding box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructurePiece {
    /// Type of piece.
    pub piece_type: PieceType,
    /// Axis-aligned bounding box in world coordinates.
    pub bounding_box: StructureBoundingBox,
    /// Rotation in 90-degree increments (0 = 0°, 1 = 90°, 2 = 180°, 3 = 270°).
    pub rotation: u8,
}

impl StructurePiece {
    /// Constructs a new `StructurePiece`.
    #[must_use]
    pub const fn new(
        piece_type: PieceType,
        bounding_box: StructureBoundingBox,
        rotation: u8,
    ) -> Self {
        Self {
            piece_type,
            bounding_box,
            rotation: rotation % 4,
        }
    }
}

/// Jigsaw structure assembler managing collision rejection and piece layouts.
#[derive(Debug, Default, Clone)]
pub struct JigsawAssembler {
    pieces: Vec<StructurePiece>,
    overall_bounds: Option<StructureBoundingBox>,
}

impl JigsawAssembler {
    /// Constructs a new empty jigsaw assembler.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pieces: Vec::new(),
            overall_bounds: None,
        }
    }

    /// Attempts to add a piece, rejecting if its bounding box collides with existing pieces.
    pub fn try_add_piece(&mut self, piece: StructurePiece) -> bool {
        for existing in &self.pieces {
            if existing.bounding_box.intersects(&piece.bounding_box) {
                return false;
            }
        }

        // Update overall bounds
        self.overall_bounds = Some(match self.overall_bounds {
            Some(b) => StructureBoundingBox::new(
                b.min_x.min(piece.bounding_box.min_x),
                b.min_y.min(piece.bounding_box.min_y),
                b.min_z.min(piece.bounding_box.min_z),
                b.max_x.max(piece.bounding_box.max_x),
                b.max_y.max(piece.bounding_box.max_y),
                b.max_z.max(piece.bounding_box.max_z),
            ),
            None => piece.bounding_box,
        });

        self.pieces.push(piece);
        true
    }

    /// Returns a slice of all placed pieces.
    #[must_use]
    pub fn pieces(&self) -> &[StructurePiece] {
        &self.pieces
    }

    /// Overall encompassing bounding box covering all placed pieces.
    #[must_use]
    pub const fn bounds(&self) -> Option<StructureBoundingBox> {
        self.overall_bounds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jigsaw_collision_avoidance() {
        let mut assembler = JigsawAssembler::new();

        let piece1 = StructurePiece::new(
            PieceType::RuinAltar,
            StructureBoundingBox::new(0, 0, 0, 10, 5, 10),
            0,
        );
        assert!(assembler.try_add_piece(piece1));

        // Overlapping piece should be rejected
        let piece2_overlapping = StructurePiece::new(
            PieceType::RuinWall,
            StructureBoundingBox::new(5, 0, 5, 15, 5, 15),
            0,
        );
        assert!(!assembler.try_add_piece(piece2_overlapping));

        // Non-overlapping piece should be accepted
        let piece3_adjacent = StructurePiece::new(
            PieceType::RuinPillar,
            StructureBoundingBox::new(12, 0, 0, 14, 5, 2),
            0,
        );
        assert!(assembler.try_add_piece(piece3_adjacent));
        assert_eq!(assembler.pieces().len(), 2);
    }
}
