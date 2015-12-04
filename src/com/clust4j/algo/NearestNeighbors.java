package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.log.Log;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;

public class NearestNeighbors extends AbstractClusterer {
	
	public static enum RunMode {
		K_NEAREST,				// Find K nearest neighbors
		RADIUS					// Return the neighborhood points within a certain radius
	}
	
	
	final public static double DEF_NEIGHBORHOOD = DBSCAN.DEF_EPS;
	final public static int DEF_K = 5;
	final public static RunMode DEF_RUN_MODE = RunMode.K_NEAREST;
	
	
	private final RunMode runmode;
	private final int k, m;
	private final double neighborhood;
	private final double[][] dist_mat;
	
	
	/** The actual nearest indices */
	volatile private ArrayList<Integer>[] nearest = null;
	
	
	
	
	public NearestNeighbors(AbstractRealMatrix data) {
		this(data, new NearestNeighborsPlanner(DEF_RUN_MODE));
	}

	public NearestNeighbors(AbstractRealMatrix data, NearestNeighborsPlanner planner) {
		super(data, planner);
		
		this.runmode = planner.runmode;
		this.k = FastMath.min(planner.k, (m = data.getRowDimension()) - 1);
		if(k <= 0)
			throw new IllegalArgumentException("k="+k);
			
		this.neighborhood = planner.neighborhood;
		
		if(verbose) {
			meta("runmode="+runmode);
			if(this.k != planner.k)
				warn(planner.k + " is greater than the number of rows in data. reducing k to " + this.k);
		}
		
		if(null == planner.dist_mat) {
			if(verbose) info("computing distance matrix (" + m + "x" + m + ")");
			dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
		} else {
			dist_mat = planner.dist_mat;
		}
	}
	
	
	

	public static class NearestNeighborsPlanner extends BaseClustererPlanner {
		private RunMode runmode = DEF_RUN_MODE;
		private int k = DEF_K;
		private double neighborhood = DEF_NEIGHBORHOOD;
		private boolean scale = DEF_SCALE;
		private Random seed = DEF_SEED;
		private GeometricallySeparable dist	= DEF_DIST;
		private boolean verbose	= DEF_VERBOSE;
		private double[][] dist_mat = null;
		
		
		public NearestNeighborsPlanner() { }
		public NearestNeighborsPlanner(RunMode runmode) {
			this.runmode = runmode;
		}
		

		@Override
		public GeometricallySeparable getSep() {
			return dist;
		}

		@Override
		public boolean getScale() {
			return scale;
		}

		@Override
		public Random getSeed() {
			return seed;
		}

		@Override
		public boolean getVerbose() {
			return verbose;
		}
		
		public NearestNeighborsPlanner setDistanceMatrix(final double[][] dist_mat) {
			this.dist_mat = MatUtils.copyMatrix(dist_mat);
			return this;
		}
		
		public NearestNeighborsPlanner setK(final int k) {
			this.k = k;
			return this;
		}
		
		public NearestNeighborsPlanner setRadius(final double d) {
			this.neighborhood = d;
			return this;
		}

		@Override
		public NearestNeighborsPlanner setScale(boolean b) {
			this.scale = b;
			return this;
		}

		@Override
		public NearestNeighborsPlanner setSeed(Random rand) {
			this.seed = rand;
			return this;
		}

		@Override
		public NearestNeighborsPlanner setVerbose(boolean b) {
			this.verbose = b;
			return this;
		}

		@Override
		public NearestNeighborsPlanner setSep(GeometricallySeparable dist) {
			this.dist = dist;
			return this;
		}
	}
	
	public int getK() {
		return k;
	}
	
	public double[][] getNearestRecords(final int row) {
		ArrayList<Integer> closest = null;
		
		try {
			closest = nearest[row];
		} catch(NullPointerException e) {
			throw new ModelNotFitException(e);
		} catch(ArrayIndexOutOfBoundsException e) {
			throw new ModelNotFitException(e);
		} catch(IndexOutOfBoundsException e) {
			throw new ModelNotFitException(e);
		}
		
		final double[][] d = new double[closest.size()][];
		for(int i = 0; i < d.length; i++) {
			d[i] = data.getRow(closest.get(i));
		}
		
		return d;
	}
	
	public double getRadius() {
		return neighborhood;
	}
	
	public RunMode getRunMode() {
		return runmode;
	}

	@Override
	public String getName() {
		return "NearestNeighbors";
	}
	
	public ArrayList<Integer>[] getNearest() {
		return nearest;
	}

	@Override
	public Algo getLoggerTag() {
		return Log.Tag.Algo.NEAREST;
	}

	@SuppressWarnings("unchecked")
	@Override
	public NearestNeighbors fit() {
		synchronized(this) {
			if(null != nearest) // synch condition
				return this;
			
			
			final boolean knn = runmode.equals(RunMode.K_NEAREST);
			if(verbose) info("identifying " + (knn ? 
				(k+" nearest records for each point") : 
					("neighborhoods for each record within radius="+neighborhood)));
			
			long start = System.currentTimeMillis();
			nearest = new ArrayList[m];
			
			
			if(knn) {
				SortedSet<Map.Entry<Integer, Double>> ordered;
				Iterator<Map.Entry<Integer, Double>> iter;
				for(int i = 0; i < m; i++) {
					nearest[i] = new ArrayList<Integer>();
					ordered = getSortedNearest(i, dist_mat);
					
					int j = 0;
					iter = ordered.iterator();
					while(j++ < k)
						nearest[i].add(iter.next().getKey());
				}
				
			} else {
				for(int i = 0; i < m - 1; i++) {
					if(null == nearest[i]) nearest[i] = new ArrayList<Integer>();
					
					for(int j = i + 1; j < m; j++) {
						if(null == nearest[j]) nearest[j] = new ArrayList<Integer>();
						
						int row = FastMath.min(i, j), col = FastMath.max(i, j);
						final double val = FastMath.abs(dist_mat[row][col]);
						
						if(val <= neighborhood) { // Then both are within eachother's neighborhood...
							nearest[i].add(j);
							nearest[j].add(i);
						}
					}
				}
			}
			
			
			if(verbose)
				info("model " + getKey() + " completed in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-start, false));
			
			return this;
		}
	}
	
	private static SortedSet<Map.Entry<Integer, Double>> getSortedNearest(final int record, final double[][] dist_mat) {
		// TM container
		TreeMap<Integer, Double> rec_to_dist = new TreeMap<Integer, Double>();
		
		final int m = dist_mat.length;
		for(int i = 0; i < m; i++) {
			if(i == record)
				continue;
			
			int row = FastMath.min(i, record), col = FastMath.max(i, record);
			rec_to_dist.put(row == record ? col : row, dist_mat[row][col]);
		}
		
		// Sort treemap on value
		// If the distance metric is a similarity metric, we want it DESC else ASC
		return ClustUtils.sortEntriesByValue( rec_to_dist );
	}
	
	
	/**
	 * For use with KNN -- doesn't take a distance matrix, calculates it on the fly
	 * @return
	 */
	protected static SortedSet<Map.Entry<Integer, Double>> getSortedNearest(final AbstractRealMatrix data, 
				final GeometricallySeparable dist, final double[] record) {
		// TM container
		TreeMap<Integer, Double> rec_to_dist = new TreeMap<Integer, Double>();
		
		// Get map of distances to each record
		for(int train_row = 0; train_row < data.getRowDimension(); train_row++) {
			final double sim = dist.getDistance(record, data.getRow(train_row));
			rec_to_dist.put(train_row, sim);
		}
		
		// Sort treemap on value
		// If the distance metric is a similarity metric, we want it DESC else ASC
		return ClustUtils.sortEntriesByValue( rec_to_dist );
	}
	

	/**
	 * For use with MeanShift so as not to create new model each iteration
	 * @param rad
	 * @return
	 */
	protected static ArrayList<Integer> getNearestWithinRadius(final double rad, final double[][] dist_mat, final int recordIdx) {
		// TM container
		final ArrayList<Integer> insideRad = new ArrayList<>();
		
		// Get map of distances to each record
		for(int train_row = 0; train_row < dist_mat.length; train_row++) {
			if(train_row == recordIdx)
				continue;
			
			final double sim = dist_mat[FastMath.min(recordIdx, train_row)][FastMath.max(recordIdx, train_row)];
			if(FastMath.abs(sim) < rad) insideRad.add(train_row);
		}
		
		// Sort treemap on value
		// If the distance metric is a similarity metric, we want it DESC else ASC
		return insideRad;
	}
}
