use hephaestus_core::{AttentionGradientViews, StridedView};
use leto::Layout;

pub(super) fn bind<'a, B>(
    query: Option<(&'a B, &'a Layout<3>)>,
    key: Option<(&'a B, &'a Layout<3>)>,
    value: Option<(&'a B, &'a Layout<3>)>,
) -> AttentionGradientViews<'a, B> {
    AttentionGradientViews {
        query: query.map(|(buffer, layout)| StridedView::new(buffer, layout)),
        key: key.map(|(buffer, layout)| StridedView::new(buffer, layout)),
        value: value.map(|(buffer, layout)| StridedView::new(buffer, layout)),
    }
}
