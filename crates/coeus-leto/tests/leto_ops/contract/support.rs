use coeus_core::{Layout, Shape};

pub(crate) fn layout(shape: &[usize]) -> Layout {
    Layout::new(Shape::from(shape.to_vec()))
}
