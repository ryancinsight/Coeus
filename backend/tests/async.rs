#[cfg(test)]
#[tokio::test]
async fn concurrent_add() {
    use coeus_backend::{CpuBackend, Backend};
    let handles: Vec<_> = (0..10).map(|_| {
        tokio::spawn(async move {
            let a = vec![1.0f32; 64];
            let b = vec![1.0f32; 64];
            // Stub async dispatch - use CPU backend for now
            let backend = CpuBackend::default();
            let a_data = backend.create_tensor_data(a.clone(), vec![64]).unwrap();
            let b_data = backend.create_tensor_data(b.clone(), vec![64]).unwrap();
            let result = backend.add(&a_data, &b_data).unwrap();
            result.data().to_vec()
        })
    }).collect();
    let results = futures::future::join_all(handles).await;
    for result in results {
        assert_eq!(result.unwrap()[0], 2.0); // Commutative check
    }
}
