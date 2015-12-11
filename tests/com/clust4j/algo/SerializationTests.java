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
		
		ap = (AffinityPropagation)AffinityPropagation.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(MatUtils.equalsExactly(a, ap.getAvailabilityMatrix()));
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
		
		agglom = (HierarchicalAgglomerative)HierarchicalAgglomerative.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(VecUtils.equalsExactly(l, agglom.getLabels()));
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
		
		db = (DBSCAN)DBSCAN.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(a == db.getNumberOfNoisePoints());
		Files.delete(path);
	}
	
	
	
	@Test
	public void testKMeans() throws FileNotFoundException, IOException, ClassNotFoundException {
		KMeans km = new KMeans(matrix,
			new KMeans.BaseKCentroidPlanner(3)
				.setScale(true)
				.setVerbose(true)).fit();
		
		final double c = km.getCost();
		km.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		km = (KMeans)KMeans.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(km.getCost() == c);
		Files.delete(path);
	}
	
	
	
	@Test
	public void testKMedoids() throws FileNotFoundException, IOException, ClassNotFoundException {
		KMedoids km = new KMedoids(matrix,
			new KMedoids.KMedoidsPlanner(3)
				.setScale(true)
				.setVerbose(true)).fit();
		
		final double c = km.getCost();
		km.saveModel(new FileOutputStream(tmpSerPath));
		assertTrue(file.exists());
		
		km = (KMedoids)KMedoids.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(km.getCost() == c);
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
		
		ms = (MeanShift)MeanShift.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(ms.getNumberOfNoisePoints() == n);
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
		
		nn = (NearestNeighbors)NearestNeighbors.loadModel(new FileInputStream(tmpSerPath));
		assertTrue(nn.getNearest()[0].equals(c));
		Files.delete(path);
	}
}
