package com.clust4j.utils.parallel;

import static org.junit.Assert.*;

import org.junit.Test;

import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class ParallelTests {

	@Test
	public void testDistributedEquals() {
		final double[] a = new double[]{1.01, 2, 3};
		final double[] b = new double[]{1, 2, 3};
		
		assertFalse( VecUtils.equalsExactlyDistributed(a, b) );
		assertTrue ( VecUtils.equalsWithToleranceDistributed(a, b, 0.011) );
	}
	
	@Test
	public void testDistributedAbs() {
		final double[] a = new double[]{-1, 2, -3, 5, -8};
		final double[] b = new double[]{1, 2, 3, 5, 8};
		assertTrue( VecUtils.equalsExactly(VecUtils.absDistributed(a), b) );
		assertFalse( VecUtils.equalsExactly(a, b) ); // Test mutability here, make sure abs didn't change anything in original...
	}
	
	@Test
	public void testDistAbsSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.absDistributed(d);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.absForceSerial(d);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed ABS test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testDistributedAdd() {
		final double[] a = new double[]{6, 1, 12,5, 1};
		final double[] b = new double[]{1, 2, 3, 5, 8};
		final double[] c = new double[]{7, 3, 15,10,9};
		final double[] d = VecUtils.copy(a);
		assertTrue( VecUtils.equalsExactly(VecUtils.addDistributed(a, b), c) );
		assertTrue( VecUtils.equalsExactly(a, d) ); // Test mutability here, make sure add didn't change anything in original...
	}
	
	@Test
	public void testDistAddSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		final double[] d2 = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.addDistributed(d,d2);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.addForceSerial(d,d2);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed ADD test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testDistributedMult() {
		final double[] a = new double[]{2, 1, 3 ,5, 1};
		final double[] b = new double[]{1, 2, 3, 5, 8};
		final double[] c = new double[]{2, 2, 9,25, 8};
		final double[] d = VecUtils.copy(a);
		assertTrue( VecUtils.equalsExactly(VecUtils.multiplyDistributed(a, b), c) );
		assertTrue( VecUtils.equalsExactly(a, d) ); // Test mutability here, make sure add didn't change anything in original...
	}
	
	@Test
	public void testDistMultSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		final double[] d2 = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.multiplyDistributed(d,d2);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.multiplyForceSerial(d,d2);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed MULT test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testDistributedLog() {
		final double[] a = new double[]{2, 2, 9,25, 8};
		assertTrue( VecUtils.equalsExactly(VecUtils.logDistributed(a), VecUtils.logForceSerial(a)) );
	}
	
	@Test
	public void testDistLogSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.logDistributed(d);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.logForceSerial(d);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed LOG test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testDistributedSub() {
		final double[] a = new double[]{2, 1, 3 ,5, 1};
		final double[] b = new double[]{1, 2, 3, 5, 8};
		final double[] c = new double[]{1,-1, 0, 0,-7};
		final double[] d = VecUtils.copy(a);
		assertTrue( VecUtils.equalsExactly(VecUtils.subtractDistributed(a, b), c) );
		assertTrue( VecUtils.equalsExactly(a, d) ); // Test mutability here, make sure add didn't change anything in original...
	}
	
	@Test
	public void testDistSubSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		final double[] d2 = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.multiplyDistributed(d,d2);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.multiplyForceSerial(d,d2);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed SUB test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testDistributedInnerProd() {
		final double[] d = new double[]{0,1};
		final double[] d2= new double[]{1,2};
		
		assertTrue(VecUtils.innerProduct(d, d2) == 2);
		assertTrue(VecUtils.innerProductDistributed(d, d2) == 2);
	}
	
	@Test
	public void testDistributedSum() {
		final double[] d = new double[]{0,1,2,3,4,5,6,7,8}; // Odd length
		final double[] d2= new double[]{1,2,3,4,5,6,7,8};   // Even length
		assertTrue(VecUtils.sumDistributed(d) == 36);
		assertTrue(VecUtils.sumDistributed(d2) == 36);
	}
	
	@Test
	public void testDistributedProd() {
		final double[] d = new double[]{0,1};
		final double[] d2= new double[]{1,2};
		assertTrue(VecUtils.prodDistributed(d) == 0);
		assertTrue(VecUtils.prodDistributed(d2) == 2);
	}
	
	@Test
	public void testDistributedNanCheck() {
		final double[] d = new double[]{0,1,Double.NaN};
		final double[] d2= new double[]{1,2};
		assertTrue(VecUtils.containsNaN(d));
		assertTrue(VecUtils.containsNaNDistributed(d));
		
		assertFalse(VecUtils.containsNaN(d2));
		assertFalse(VecUtils.containsNaNDistributed(d2));
	}
	
	@Test
	public void testDistributedNanCount() {
		final double[] d = new double[]{0,1,Double.NaN};
		final double[] d2= new double[]{1,2};
		assertTrue(VecUtils.nanCount(d) == 1);
		assertTrue(VecUtils.nanCountDistributed(d) == 1);
		
		assertTrue(VecUtils.nanCount(d2) == 0);
		assertTrue(VecUtils.nanCountDistributed(d2) == 0);
	}
	
	@Test
	public void testDistSumAccuracy() {
		final double[] d = VecUtils.randomGaussian(500000, 1);
		final double distSum = VecUtils.sumDistributed(d);
		final double sum = VecUtils.sum(d);
		assertTrue(distSum == sum);
	}
	
	@Test
	public void testDistInnerProductAccuracy() {
		final double[] d = VecUtils.randomGaussian(500000, 1);
		final double[] d2 = VecUtils.randomGaussian(500000, 1);
		assertFalse(VecUtils.equalsExactly(d, d2));
		
		final double distInner = VecUtils.innerProductDistributed(d, d2);
		final double inner = VecUtils.innerProduct(d, d2);
		assertTrue(distInner == inner);
	}
	
	@Test
	public void testDistProdAccuracy() {
		final double[] d = VecUtils.randomGaussian(500_000, 1);
		final double distProd = VecUtils.prodDistributed(d);
		final double prod = VecUtils.prod(d);
		assertTrue(distProd == prod);
	}
	
	@Test
	public void testDistNanSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.nanCountDistributed(d);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.nanCountForceSerial(d);
		final long nanTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed NaN test:\tDist: " + distTime + ", Normal: " + nanTime);
	}
	
	@Test
	public void testDistSumSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.sumDistributed(d);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.sumForceSerial(d);
		final long sumTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed SUM test:\tDist: " + distTime + ", Normal: " + sumTime);
	}
	
	@Test
	public void testDistProdSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.prodDistributed(d);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.prodForceSerial(d);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed PROD test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testDistInnerProdSpeed() {
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		final double[] d2 = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		VecUtils.innerProductDistributed(d,d2);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		VecUtils.innerProductForceSerial(d,d2);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed INNER test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void loadTestDist() {
		final int times= 50;
		final double[] d = VecUtils.randomGaussian(9_000_000, 1);
		final double[] d2 = VecUtils.randomGaussian(9_000_000, 1);
		
		long start = System.currentTimeMillis();
		for(int i = 0; i < times; i++);
			VecUtils.innerProductDistributed(d,d2);
		final long distTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		for(int i = 0; i < times; i++);
			VecUtils.innerProductForceSerial(d,d2);
		final long prodTime = System.currentTimeMillis() - start;
		
		System.out.println("Distributed LOAD test:\tDist: " + distTime + ", Normal: " + prodTime);
	}
	
	@Test
	public void testMult() {
		final double[][] a = MatUtils.randomGaussian(1000, 20);
		final double[][] b = MatUtils.randomGaussian(20, 6000);
		
		long start = System.currentTimeMillis();
		final double[][] ca = MatUtils.multiplyForceSerial(a, b);
		long serialTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		final double[][] cb = MatUtils.multiplyDistributed(a, b);
		long paraTime = System.currentTimeMillis() - start;
		
		assertTrue(MatUtils.equalsExactly(ca, cb));
		System.out.println("Dist MatMult test:\tParallel="+paraTime+", Serial="+serialTime);
	}

	@Test
	public void testMultRealBig() {
		final double[][] a = MatUtils.randomGaussian(5000, 20);
		final double[][] b = MatUtils.randomGaussian(20, 6000);
		
		long start = System.currentTimeMillis();
		final double[][] ca = MatUtils.multiplyForceSerial(a, b);
		long serialTime = System.currentTimeMillis() - start;
		
		start = System.currentTimeMillis();
		final double[][] cb = MatUtils.multiplyDistributed(a, b);
		long paraTime = System.currentTimeMillis() - start;
		
		assertTrue(MatUtils.equalsWithTolerance(ca, cb, 1e-8));
		System.out.println("Dist MatMult test:\tParallel="+paraTime+", Serial="+serialTime);
	}
}
