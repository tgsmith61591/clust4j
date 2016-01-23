package com.clust4j.algo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeMap;

import org.apache.commons.math3.linear.AbstractRealMatrix;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.algo.preprocess.FeatureNormalization;
import com.clust4j.log.Log;
import com.clust4j.log.LogTimeFormatter;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.utils.ClustUtils;
import com.clust4j.utils.GeometricallySeparable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.ModelNotFitException;
import com.clust4j.utils.VecUtils;

/**
 * A generalized {@link AbstractClusterer} that fits the nearest neighbor
 * records for each record in an {@link AbstractRealMatrix}. This algorithm can
 * be run in two modes ({@link RunMode}):
 * <p>
 * <b>K_NEAREST</b>: will find the K nearest neighbors for each record in the matrix. In the
 * case of <i>k</i> exceeding or equaling the number of rows in the data, <i>k</i> will
 * be truncated to <i>m - 1</i>, where <i>m</i> is the number of records. Default = 5.
 * 
 * <p>
 * <b>RADIUS</b>: will identify the records within a radius of each point. Note that the
 * number of points will vary for each record; some may even result in zero neighbors. Default = 0.5.
 * 
 * <p>
 * This algorithm is used extensively internally within various algorithms including {@link MeanShift},
 * {@link DBSCAN}, and {@link KNN}.
 * 
 * @author Taylor G Smith &lt;tgsmith61591@gmail.com&gt;
 * @see {@link RunMode}
 */
public class NearestNeighbors extends AbstractClusterer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -1261837566319096096L;


	/**
	 * The mode in which to run the {@link NearestNeighbors} algorithm:
	 * <p>
	 * <b>K_NEAREST</b>: will find the K nearest neighbors for each record in the matrix. In the
	 * case of <i>k</i> exceeding or equaling the number of rows in the data, <i>k</i> will
	 * be truncated to <i>m - 1</i>, where <i>m</i> is the number of records. Default = 5.
	 * 
	 * <p>
	 * <b>RADIUS</b>: will identify the records within a radius of each point. Note that the
	 * number of points will vary for each record; some may even result in zero neighbors. Default = 0.5.
	 * 
	 * @author Taylor G Smith
	 */
	public static enum RunMode implements java.io.Serializable {
		/** Find K nearest neighbors*/
		K_NEAREST,
		
		/** Return the neighborhood points within a certain radius -- can be zero */
		RADIUS
	}
	
	
	final public static double DEF_EPS_RADIUS = DBSCAN.DEF_EPS;
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
		final boolean radius_run = runmode.equals(RunMode.RADIUS);
		
		this.k = FastMath.min(planner.k, (m = data.getRowDimension()) - 1);
		if(k <= 0 && !radius_run)
			throw new IllegalArgumentException("k="+k);
			
		this.neighborhood = planner.neighborhood;
		
		meta("runmode="+runmode);
		meta(radius_run?("radius="+neighborhood):("k="+k));
		
		if(this.k != planner.k && !radius_run) {
			warn("provided k (" + planner.k + ") is greater than the number of "
				+ "rows in data. reducing k to " + this.k + " (m - 1)");
		}
		
		
		// Check whether using similarity metric AND radius...
		// Will throw a warning if so.
		if(radius_run) AbstractDensityClusterer.checkState(this);
		
		
		if(null == planner.dist_mat) {
			info("computing distance matrix (" + m + "x" + m + ")");
			dist_mat = ClustUtils.distanceUpperTriangMatrix(data, getSeparabilityMetric());
		} else {
			dist_mat = planner.dist_mat;
		}
	}
	
	
	

	public static class NearestNeighborsPlanner extends BaseClustererPlanner {
		private RunMode runmode = DEF_RUN_MODE;
		private int k = DEF_K;
		private FeatureNormalization norm = DEF_NORMALIZER;
		private double neighborhood = DEF_EPS_RADIUS;
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
		public NearestNeighbors buildNewModelInstance(AbstractRealMatrix data) {
			return new NearestNeighbors(data, this);
		}
		
		@Override
		public NearestNeighborsPlanner copy() {
			return new NearestNeighborsPlanner(runmode)
				.setK(k)
				.setRadius(neighborhood)
				.setScale(scale)
				.setSeed(seed)
				.setSep(dist)
				.setVerbose(verbose)
				.setDistanceMatrix(dist_mat)
				.setNormalizer(norm);
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
			this.dist_mat = null == dist_mat ? null : MatUtils.copy(dist_mat);
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
		
		public NearestNeighborsPlanner setRunMode(final RunMode mode) {
			this.runmode = mode;
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
		
		@Override
		public FeatureNormalization getNormalizer() {
			return norm;
		}
		@Override
		public NearestNeighborsPlanner setNormalizer(FeatureNormalization norm) {
			this.norm = norm;
			return this;
		}
	}
	
	public int getK() {
		return k;
	}
	
	/**
	 * Return the rows that correspond to the nearest neighbor
	 * indices identified in the {@link #fit()} method.
	 * @param rowNumber
	 * @throws ModelNotFitException if the model is not fit yet
	 * @return the nearest records
	 */
	public double[][] getNearestRecords(final int rowNumber) {
		ArrayList<Integer> closest = null;
		
		try {
			closest = nearest[rowNumber];
		} catch(NullPointerException e) {
			throw new ModelNotFitException(e);
		} catch(ArrayIndexOutOfBoundsException e) {
			throw new ModelNotFitException(e);
		} catch(IndexOutOfBoundsException e) {
			throw new ModelNotFitException(e);
		}
		
		final double[][] d = new double[closest.size()][];
		for(int i = 0; i < d.length; i++) {
			d[i] = VecUtils.copy(data.getRow(closest.get(i)));
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
		try {
			@SuppressWarnings("unchecked")
			final ArrayList<Integer>[] copy = new ArrayList[nearest.length];
			
			int i = 0;
			for(ArrayList<Integer> ai: nearest)
				copy[i++] = VecUtils.copy(ai);
			
			return copy;
		} catch(NullPointerException e) {
			String error = "model has not yet been fit";
			error(error);
			throw new ModelNotFitException(error);
		}
	}

	@Override
	public Algo getLoggerTag() {
		return Log.Tag.Algo.NEAREST;
	}

	@SuppressWarnings("unchecked")
	@Override
	public NearestNeighbors fit() {
		synchronized(this) {
			
			try {
				if(null != nearest) // synch condition
					return this;
				
				
				final boolean knn = runmode.equals(RunMode.K_NEAREST);
				info("identifying " + (knn ? 
					(k+" nearest record"+(k!=1?"s":"")+" for each point") : 
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
					
					// Check how many weren't classified
					int ct = 0;
					for(int i = 0; i < m; i++) if(nearest[i].isEmpty()) ct++;
					if(ct > 0) {
						warn(ct + " record" + (ct!=1?"s have":" has") + 
							" no records within radius=" + neighborhood);
					}
				}
				
				
				info("model " + getKey() + " completed in " + 
					LogTimeFormatter.millis(System.currentTimeMillis()-start, false) + 
					System.lineSeparator());
				
				return this;
			} catch(OutOfMemoryError | StackOverflowError e) {
				error(e.getLocalizedMessage() + " - ran out of memory during model fitting");
				throw e;
			} // end try/catch
			
		} // end synch
	} // end fit
	
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
	
	
	public static int[] getKNearest(final double[] record, 
			final double[][] matrix, int k, final GeometricallySeparable sep) {
		// TM container
		TreeMap<Integer, Double> rec_to_dist = new TreeMap<>();
		k = FastMath.min(k, matrix.length);
		
		
		if(matrix.length == 0)
			throw new IllegalArgumentException("empty matrix");
		if(k < 1)
			throw new IllegalArgumentException("illegal k value");
		
		
		for(int i = 0; i < matrix.length; i++)
			rec_to_dist.put(i, sep.getDistance(record, matrix[i]));
		
		
		int rec = 0;
		final int[] nrst = new int[k];
		SortedSet<Map.Entry<Integer, Double>> ordered = ClustUtils.sortEntriesByValue(rec_to_dist);
		for(Map.Entry<Integer, Double> entry: ordered) {
			if(rec == k)
				break;
			nrst[rec++] = entry.getKey();
		}
		
		return nrst;
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
