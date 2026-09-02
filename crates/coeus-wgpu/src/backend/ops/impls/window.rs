pub(super) struct WindowConfiguration<const S: usize> {
    pub(super) kernel: [usize; S],
    pub(super) stride: [usize; S],
    pub(super) padding: [usize; S],
    pub(super) dilation: [usize; S],
}
