package com.clust4j.algo;

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.junit.Test;

import com.clust4j.algo.MeanShift.MeanShiftPlanner;
import com.clust4j.algo.NearestNeighbors.NearestNeighborsPlanner;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;

public class SerializationTests {
	private static Array2DRowRealMatrix matrix = ClustTests.getRandom(250, 10);
	private static String tmpSerPath = "model.ser";
	private static File file = new File(tmpSerPath);
	private static Path path = FileSystems.getDefault().getPath(tmpSerPath);

	
	
	@Test
	public void testAffinity() throws FileNotFoundException, IOException, ClassNotFoundException {
		AffinityPropagation ap = new AffinityPropagation(matrix, 
			new AffinityPropagation
				.AffinityPropagationPlanner()
					.setVerbose(true)).fit();
		
		double[][] a = ap.getAvailabilityMatrix();
		ap.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		AffinityPropagation ap2 = (AffinityPropagation)AffinityPropagation.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(MatUtils.equalsExactly(a, ap2.getAvailabilityMatrix()));
		assertTrue(ap2.equals(ap));
		Files.delete(path);
	}

	
	
	@Test
	public void testAgglomerative() throws FileNotFoundException, IOException, ClassNotFoundException {
		HierarchicalAgglomerative agglom = 
			new HierarchicalAgglomerative(matrix, 
				new HierarchicalAgglomerative.HierarchicalPlanner()
					.setScale(true)
					.setVerbose(true)).fit();
		
		final int[] l = agglom.getLabels();
		agglom.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		HierarchicalAgglomerative agglom2 = (HierarchicalAgglomerative)HierarchicalAgglomerative.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(VecUtils.equalsExactly(l, agglom2.getLabels()));
		assertTrue(agglom2.equals(agglom));
		Files.delete(path);
	}
	
	
	
	@Test
	public void testDBSCAN() throws FileNotFoundException, IOException, ClassNotFoundException {
		DBSCAN db = new DBSCAN(matrix, 
			new DBSCAN.DBSCANPlanner(0.75)
				.setScale(true)
				.setMinPts(1)
				.setVerbose(true)).fit();
		
		int a = db.getNumberOfNoisePoints();
		db.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		DBSCAN db2 = (DBSCAN)DBSCAN.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(a == db2.getNumberOfNoisePoints());
		assertTrue(db.equals(db2));
		Files.delete(path);
	}
	
	
	
	@Test
	public void testKMeans() throws FileNotFoundException, IOException, ClassNotFoundException {
		KMeans km = new KMeans(matrix,
			new KMeans.KMeansPlanner(3)
				.setScale(true)
				.setVerbose(true)).fit();
		
		final double c = km.totalCost();
		km.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		KMeans km2 = (KMeans)KMeans.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(km2.totalCost() == c);
		assertTrue(km.equals(km2));
		Files.delete(path);
	}
	
	
	
	@Test
	public void testKMedoids() throws FileNotFoundException, IOException, ClassNotFoundException {
		KMedoids km = new KMedoids(matrix,
			new KMedoids.KMedoidsPlanner(3)
				.setScale(true)
				.setVerbose(true)).fit();
		
		final double c = km.totalCost();
		km.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		KMedoids km2 = (KMedoids)KMedoids.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(km2.totalCost() == c);
		assertTrue(km2.equals(km));
		Files.delete(path);
	}
	
	
	
	@Test
	public void testMeanShift() throws FileNotFoundException, IOException, ClassNotFoundException {
		MeanShift ms = new MeanShift(matrix,
			new MeanShiftPlanner(0.5)
				.setVerbose(true)).fit();
		
		final double n = ms.getNumberOfNoisePoints();
		ms.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		MeanShift ms2 = (MeanShift)MeanShift.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(ms2.getNumberOfNoisePoints() == n);
		assertTrue(ms.equals(ms2));
		Files.delete(path);
	}
	
	
	
	@Test
	public void testNN() throws FileNotFoundException, IOException, ClassNotFoundException {
		NearestNeighbors nn = new NearestNeighbors(matrix, 
			new NearestNeighborsPlanner()
				.setK(5)
				.setVerbose(true)
				.setScale(true)).fit();
		
		final ArrayList<Integer> c = nn.getNearest()[0];
		nn.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		NearestNeighbors nn2 = (NearestNeighbors)NearestNeighbors.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(nn2.getNearest()[0].equals(c));
		assertTrue(nn2.equals(nn));
		Files.delete(path);
	}
	
	@Test
	public void testHDBSCAN() throws FileNotFoundException, IOException, ClassNotFoundException {
		HDBSCAN hd = new HDBSCAN(matrix, 
			new HDBSCAN.HDBSCANPlanner(1)
				.setVerbose(true)
				.setScale(true)).fit();

		final int[] labels = hd.getLabels();
		hd.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		HDBSCAN hd2 = (HDBSCAN)HDBSCAN.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(VecUtils.equalsExactly(hd2.getLabels(), labels));
		assertTrue(hd.equals(hd2));
		Files.delete(path);
	}
}
