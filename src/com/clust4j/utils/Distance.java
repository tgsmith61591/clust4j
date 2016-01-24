package com.clust4j.utils;

import org.apache.commons.math3.util.FastMath;

public enum Distance implements GeometricallySeparable, java.io.Serializable {
	
	HAMMING {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a, b);
			
			final int n = a.length;
			double ct = 0;
			for(int i = 0; i < n; i++)
				if(a[i] != b[i])
					ct++;
			
			return ct / n;
		}
		
		@Override
		public String getName() {
			return "Hamming";
		}
	},
	
	MANHATTAN {
		
		@Override 
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				double diff = a[i] - b[i];
				sum += FastMath.abs(diff);
			}
			
			return sum;
		}
		
		@Override
		public String getName() {
			return "Manhattan";
		}
	},
	
	
	
	EUCLIDEAN {
		
		@Override 
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			double sum = 0;
			for(int i = 0; i < a.length; i++) {
				// Don't use math.pow -- too expensive
				double diff = a[i]-b[i];
				sum += diff * diff;
			}
			
			return FastMath.sqrt(sum);
		}
		
		@Override
		public String getName() {
			return "Euclidean";
		}
	},
	
	
	BRAY_CURTIS {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			final int n = a.length;
			double sum_1 = 0, sum_2 = 0;
			for(int i = 0; i < n; i++) {
				sum_1 += FastMath.abs(a[i] - b[i]);
				sum_2 += FastMath.abs(a[i] + b[i]);
			}
			
			return sum_1 / sum_2;
		}
		
		@Override
		public String getName() {
			return "BrayCurtis";
		}
	},
	
	
	CANBERRA {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a,b);
			
			final int n = a.length;
			double sum=0;
			for(int i = 0; i < n; i++) {
				sum += ( FastMath.abs(a[i] - b[i]) /
					(FastMath.abs(a[i]) + FastMath.abs(b[i])) );
			}
			
			return sum;
		}
		
		@Override
		public String getName() {
			return "Canberra";
		}
	},
	
	
	
	CHEBYSHEV {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			VecUtils.checkDims(a, b);
			
			final int n = a.length;
			double max = 0;
			for(int i = 0; i < n; i++) {
				double abs = FastMath.abs(a[i] - b[i]);
				if(abs > max)
					max = abs;
			}
			
			return max;
		}
		
		@Override
		public String getName() {
			return "Chebyshev";
		}
	},
	
	
	DICE {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			TriTup<Double, Double, Double> corr = 
				BooleanCorrespondence.correspondFtTfTt(a, b);
			double nft_sum = corr.one, ntf_sum = corr.two, ntt_sum = corr.three;
			return (ntf_sum + nft_sum) / (2.0 * ntt_sum + ntf_sum + nft_sum);
		}
		
		@Override
		public String getName() {
			return "Dice";
		}
	},
	
	
	KULSINSKI {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			QuadTup<Double, Double, Double, Double> correspondence = 
				BooleanCorrespondence.correspondAll(a, b);
			double nft_sum = correspondence.two, 
				ntf_sum = correspondence.three, ntt_sum = correspondence.four;
			
			final int n = a.length;
			return (ntf_sum + nft_sum - ntt_sum + n) / (ntf_sum + nft_sum + n);
		}
		
		@Override
		public String getName() {
			return "Kulsinski";
		}
	},
	
	
	ROGERS_TANIMOTO {
		
		@Override
		public double getDistance(final double[]a, final double[] b) {
			QuadTup<Double, Double, Double, Double> correspondence = 
					BooleanCorrespondence.correspondAll(a, b);
			double nff_sum = correspondence.one, nft_sum = correspondence.two, 
				   ntf_sum = correspondence.three, ntt_sum = correspondence.four;
			
			return (2.0 * (ntf_sum + nft_sum)) / (ntt_sum + nff_sum + (2.0 * (ntf_sum + nft_sum)));
		}
		
		@Override
		public String getName() {
			return "RogersTanimoto";
		}
	},
	
	
	RUSSELL_RAO {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			final double ip = VecUtils.innerProduct(a, b);
			final int n = a.length;
			return (n - ip) / n;
		}
		
		@Override
		public String getName() {
			return "RussellRao";
		}
	},
	
	
	SOKAL_SNEATH {
		
		@Override
		public double getDistance(final double[] a, final double[] b) {
			TriTup<Double, Double, Double> corr = 
				BooleanCorrespondence.correspondFtTfTt(a, b);
			double nft_sum = corr.one, ntf_sum = corr.two, ntt_sum = corr.three;
			double denom = ntt_sum + 2.0 * (ntf_sum + nft_sum);
			
			if(0 == denom)
				throw new ArithmeticException("Undefined result for Sokal-Sneath distance");
			
			return (2.0 * (nft_sum + ntf_sum)) / denom;
		}
		
		@Override
		public String getName() {
			return "SokalSneath";
		}
	},
	
	
	YULE {
		
		@Override
		public double getDistance(final double[]a, final double[] b) {
			QuadTup<Double, Double, Double, Double> correspondence = 
					BooleanCorrespondence.correspondAll(a, b);
			double nff_sum = correspondence.one, nft_sum = correspondence.two, 
				   ntf_sum = correspondence.three, ntt_sum = correspondence.four;
			
			return (2.0 * ntf_sum * nft_sum) / (ntt_sum * nff_sum + ntf_sum * nft_sum);
		}
		
		@Override
		public String getName() {
			return "Yule";
		}
	}
}
