package com.clust4j.utils;

class BooleanCorrespondence {
	
	static QuadTup<Double, Double, Double, Double> correspondAll(final double[]a, final double[]b) {
		VecUtils.checkDims(a, b);
		
		final int n = a.length;
		double nff_sum = 0, nft_sum = 0, ntf_sum = 0, ntt_sum = 0;
		double not_a, not_b;
		
		for(int i = 0; i < n; i++) {
			not_a = 1.0 - a[i];
			not_b = 1.0 - b[i];
			
			nff_sum += (not_a * not_b);
			nft_sum += (not_a * b[i]);
			ntf_sum += (a[i] * not_b);
			ntt_sum += (a[i] * b[i]);
		}
		
		return new QuadTup<>(nff_sum, nft_sum, ntf_sum, ntt_sum);
	}
	
	static EntryPair<Double, Double> correspondFtTf(final double[] a, final double[] b) {
		VecUtils.checkDims(a, b);
		
		final int n = a.length;
		double nft_sum = 0, ntf_sum = 0;
		double not_a, not_b;
		
		for(int i = 0; i < n; i++) {
			not_a = 1.0 - a[i];
			not_b = 1.0 - b[i];
			
			nft_sum += (not_a * b[i]);
			ntf_sum += (a[i] * not_b);
		}
		
		return new EntryPair<>(nft_sum, ntf_sum);
	}
	
	static TriTup<Double, Double, Double> correspondFtTfTt(final double[] a, final double[] b) {
		VecUtils.checkDims(a, b);
		
		final int n = a.length;
		double nft_sum = 0, ntf_sum = 0, ntt_sum = 0;
		double not_a, not_b;
		
		for(int i = 0; i < n; i++) {
			not_a = 1.0 - a[i];
			not_b = 1.0 - b[i];
			
			nft_sum += (not_a * b[i]);
			ntf_sum += (a[i] * not_b);
			ntt_sum += a[i] * b[i];
		}
		
		return new TriTup<>(nft_sum, ntf_sum, ntt_sum);
	}
}
