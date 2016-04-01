package com.clust4j.algo;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.Precision;
import org.junit.Test;

import com.clust4j.algo.BallTree;
import com.clust4j.algo.KDTree;
import com.clust4j.algo.NearestNeighborHeapSearch.Heap;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeHeap.NodeHeapData;
import com.clust4j.algo.NearestNeighborHeapSearch.MutableDouble;
import com.clust4j.algo.NearestNeighborHeapSearch.NeighborsHeap;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeData;
import com.clust4j.algo.NearestNeighborHeapSearch.NodeHeap;
import com.clust4j.algo.NearestNeighborHeapSearch.PartialKernelDensity;
import com.clust4j.algo.NearestNeighborHeapSearch.Neighborhood;
import com.clust4j.data.ExampleDataSets;
import com.clust4j.log.Loggable;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.DistanceMetric;
import com.clust4j.metrics.pairwise.HaversineDistance;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.QuadTup;
import com.clust4j.utils.TriTup;
import com.clust4j.utils.VecUtils;
import com.clust4j.utils.Series.Inequality;
import com.clust4j.utils.VecUtils.VecSeries;

public class NNHSTests {
	final public static Array2DRowRealMatrix IRIS = ExampleDataSets.loadIris().getData();
	
	final static double[][] a = new double[][]{
		new double[]{0,1,0,2},
		new double[]{0,0,1,2},
		new double[]{5,6,7,4}
	};

	@Test
	public void testKD1() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		KDTree kd = new KDTree(mat);
		
		QuadTup<double[][], int[], NodeData[], double[][][]> arrays = kd.getArrays();
		
		assertTrue(MatUtils.equalsExactly(arrays.one, a));
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2}, arrays.two));
		
		TriTup<Integer, Integer, Integer> stats = kd.getTreeStats();
		assertTrue(stats.one == 0);
		assertTrue(stats.two == 0);
		assertTrue(stats.three==0);
		
		NodeData data = arrays.three[0];
		assertTrue(data.idx_start == 0);
		assertTrue(data.idx_end == 3);
		assertTrue(data.is_leaf);
		assertTrue(data.radius == 1);
	}
	
	@Test
	public void testBall1() {
		final Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		BallTree ball = new BallTree(mat);
		
		QuadTup<double[][], int[], NodeData[], double[][][]> arrays = ball.getArrays();
		
		assertTrue(MatUtils.equalsExactly(arrays.one, a));
		assertTrue(VecUtils.equalsExactly(new int[]{0,1,2}, arrays.two));
		
		TriTup<Integer, Integer, Integer> stats = ball.getTreeStats();
		assertTrue(stats.one == 0);
		assertTrue(stats.two == 0);
		assertTrue(stats.three==0);
		
		NodeData data = arrays.three[0];
		assertTrue(data.idx_start == 0);
		assertTrue(data.idx_end == 3);
		assertTrue(data.is_leaf);
		assertTrue(data.radius == 6.716480559869961);
		
		double[][][] trip = arrays.four;
		assertTrue(trip.length == 1);
		assertTrue(trip[0][0][0] == 1.6666666666666667);
		assertTrue(trip[0][0][1] == 2.3333333333333333);
		assertTrue(trip[0][0][2] == 2.6666666666666667);
		assertTrue(trip[0][0][3] == 2.6666666666666667);
	}

	
	@Test
	public void testKernelDensitiesAndNorms() {
		/*
		 * These are the numbers the sklearn code produces (though numpy rounds up more than java)
		 */
		
		// Test where dist > h first
		double dist = 5.0, h = 1.3;
		
		assertTrue(PartialKernelDensity.LOG_GAUSSIAN.getDensity(dist, h) == -7.396449704142011);
		assertTrue(PartialKernelDensity.LOG_TOPHAT.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_EPANECHNIKOV.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_EXPONENTIAL.getDensity(dist, h) == -3.846153846153846);
		assertTrue(PartialKernelDensity.LOG_LINEAR.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		assertTrue(PartialKernelDensity.LOG_COSINE.getDensity(dist, h) == Double.NEGATIVE_INFINITY);
		
		// Test where dist < h second
		dist = 1.3; 
		h = 5.0;
		
		assertTrue(PartialKernelDensity.LOG_GAUSSIAN.getDensity(dist, h) == -0.033800000000000004);
		assertTrue(PartialKernelDensity.LOG_TOPHAT.getDensity(dist, h) == 0.0);
		assertTrue(PartialKernelDensity.LOG_EPANECHNIKOV.getDensity(dist, h) == -0.06999337182053497);
		assertTrue(PartialKernelDensity.LOG_EXPONENTIAL.getDensity(dist, h) == -0.26);
		assertTrue(PartialKernelDensity.LOG_LINEAR.getDensity(dist, h) == -0.3011050927839216);
		assertTrue(PartialKernelDensity.LOG_COSINE.getDensity(dist, h) == -0.08582521637384073);
		
		
		/*
		 * Now test Kernel norms...
		 */
		h = 1.3;
		int d = 5;
		
		assertTrue(NearestNeighborHeapSearch.logKernelNorm(h, d, PartialKernelDensity.LOG_GAUSSIAN) == -5.906513988360818);
		assertTrue(NearestNeighborHeapSearch.logKernelNorm(h, d, PartialKernelDensity.LOG_TOPHAT) == -2.972672434613881);
		assertTrue(NearestNeighborHeapSearch.logKernelNorm(h, d, PartialKernelDensity.LOG_EPANECHNIKOV) == -1.7199094661185133);
		assertTrue(NearestNeighborHeapSearch.logKernelNorm(h, d, PartialKernelDensity.LOG_EXPONENTIAL) == -7.760164177395928);
		assertTrue(NearestNeighborHeapSearch.logKernelNorm(h, d, PartialKernelDensity.LOG_LINEAR) == -1.1809129653858264);
		assertTrue(NearestNeighborHeapSearch.logKernelNorm(h, d, PartialKernelDensity.LOG_COSINE) == -1.588674327991151);
	}
	
	@Test
	public void testEstimateKernelDensity() {
		final double h = 0.5, at = 0.0, rt = 1e-8;
		final KDTree k = new KDTree(IRIS);
		
		/*
		 *  GAUSSIAN
		 */
		double[] exp = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_GAUSSIAN, at, rt, false);
		double[] log = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_GAUSSIAN, at, rt, true);
		double[] expected_exp = new double[]{
			12.28012213,  10.45762713,  10.79173934,   9.91100044,
	        11.83821592,   7.37523119,  10.15806662,  12.78705716,
	         6.73568504,  11.16047533,   9.59242696,  11.78397871,
	        10.05861109,   5.13144797,   3.79949718,   2.7569215 ,
	         7.40977567,  12.2664965 ,   5.67837233,  10.22914483,
	         9.25843299,  10.81676261,   6.22015249,  10.04296168,
	         8.6977775 ,  10.0373372 ,  11.93847251,  11.9609578 ,
	        11.85992468,  10.87558964,  10.90385856,   9.69118458,
	         6.39025706,   4.89462441,  11.16047533,  10.66063678,
	         8.4050554 ,  11.16047533,   7.16013574,  12.57084638,
	        11.81003215,   2.28644394,   7.79668271,   9.71624306,
	         7.34342451,  10.21286369,   9.97460948,  10.33002923,
	        10.38942623,  12.55079235,   4.49638551,   8.32967062,
	         6.11506094,   5.93506098,   9.26420516,   9.14402071,
	         8.32269348,   2.2884424 ,   7.84879076,   5.06544992,
	         2.1216892 ,   8.87592108,   4.46667403,  10.6051664 ,
	         4.99393933,   6.30282393,   8.16600372,   7.6139944 ,
	         5.50766354,   6.93374452,   8.06491113,   7.72412736,
	         8.39118297,   9.05573515,   8.0914415 ,   7.38763938,
	         6.64913168,   8.64158676,  10.68749816,   4.44914895,
	         5.90974638,   5.2120019 ,   7.77505348,   8.99693593,
	         6.34485692,   6.61439069,   7.6741988 ,   5.84235461,
	         7.6827928 ,   7.16366372,   7.25664652,  10.39958075,
	         8.17091265,   2.4916162 ,   8.7106951 ,   8.17098773,
	         8.98842272,   9.23702581,   1.86565956,   8.88946089,
	         3.52157191,   7.4583668 ,   5.2160347 ,   7.55914067,
	         6.82334224,   2.29047669,   2.06620082,   2.98708186,
	         4.23827541,   2.29040748,   8.65940973,   8.90651813,
	         7.86138621,   5.52498516,   4.34094775,   7.39080672,
	         8.51477518,   1.15631644,   1.23170302,   5.02624612,
	         6.22942732,   5.97238892,   1.87082373,  10.01809726,
	         6.82327907,   3.89462823,  10.46733863,  10.30262734,
	         7.61239032,   3.63114801,   3.19895026,   1.15852059,
	         7.23468987,   9.47646535,   4.19472305,   2.29812615,
	         5.17062191,   8.23661115,   9.89631121,   7.19670567,
	         6.7522757 ,   5.4892544 ,   7.4583668 ,   5.75394048,
	         5.42296292,   7.0839312 ,   7.93392041,   9.57929288,
	         5.50956848,   8.60444823
		};
		
		double[] expected_log = new double[]{
			2.50798187,  2.34733158,  2.37878097,  2.2936453 ,  2.47133294,
	        1.99812725,  2.31826813,  2.5484335 ,  1.90741952,  2.41237855,
	        2.26097393,  2.46674087,  2.30842909,  1.63538788,  1.33486874,
	        1.01411466,  2.00280016,  2.50687168,  1.73666463,  2.32524098,
	        2.22553481,  2.38109702,  1.82779442,  2.30687206,  2.16306753,
	        2.30631186,  2.47976617,  2.48164783,  2.47316504,  2.38652079,
	        2.38911672,  2.27121667,  1.8547745 ,  1.58813754,  2.41237855,
	        2.36655815,  2.12883336,  2.41237855,  1.96852894,  2.53138035,
	        2.46894935,  0.82699775,  2.05369835,  2.27379903,  1.99380529,
	        2.32364807,  2.30004281,  2.33505511,  2.34078858,  2.5297838 ,
	        1.50327385,  2.11982391,  1.81075473,  1.7808773 ,  2.22615807,
	        2.21310019,  2.11898594,  0.82787141,  2.06035948,  1.62244296,
	        0.75221256,  2.18334211,  1.49664407,  2.36134128,  1.60822504,
	        1.84099778,  2.09997965,  2.02998792,  1.70614049,  1.9364    ,
	        2.08752269,  2.04434885,  2.12718151,  2.20339828,  2.0908069 ,
	        1.99980825,  1.89448627,  2.15658622,  2.36907466,  1.49271283,
	        1.77660292,  1.65096402,  2.05092034,  2.19688407,  1.84764455,
	        1.88924768,  2.0378639 ,  1.7651339 ,  2.03898313,  1.96902154,
	        1.98191781,  2.34176549,  2.10058061,  0.91293158,  2.16455159,
	        2.1005898 ,  2.19593738,  2.22321995,  0.62361464,  2.1848664 ,
	        1.25890746,  2.00933646,  1.65173748,  2.02275752,  1.92034942,
	        0.82875996,  0.72571157,  1.09429695,  1.44415644,  0.82872974,
	        2.15864656,  2.18678338,  2.06196295,  1.70928056,  1.4680927 ,
	        2.00023689,  2.14180291,  0.14523947,  0.20839778,  1.61467341,
	        1.82928441,  1.787147  ,  0.62637883,  2.30439318,  1.92034016,
	        1.35959823,  2.3482598 ,  2.33239894,  2.02977723,  1.28954886,
	        1.16282271,  0.14714384,  1.97888749,  2.24881139,  1.43382732,
	        0.83209407,  1.64299297,  2.10858899,  2.29216208,  1.97362338,
	        1.90987959,  1.70279244,  2.00933646,  1.74988492,  1.69064233,
	        1.95782901,  2.07114729,  2.25960378,  1.7064863 ,  2.15227931
		};
		
		assertTrue(VecUtils.equalsWithTolerance(exp, expected_exp, 1e-8));
		assertTrue(VecUtils.equalsWithTolerance(log, expected_log, 1e-8));
		
		
		/*
		 * TOPHAT
		 */
		exp = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_TOPHAT, at, rt, false);
		log = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_TOPHAT, at, rt, true);
		expected_exp = new double[]{
			 81.05694691,   61.60327965,   71.33011328,   58.36100178,
	         61.60327965,   35.66505664,   64.84555753,  106.99516993,
	         35.66505664,   74.57239116,   58.36100178,   97.2683363 ,
	         64.84555753,   16.21138938,    9.72683363,    6.48455575,
	         35.66505664,   81.05694691,   12.96911151,   61.60327965,
	         42.1496124 ,   71.33011328,    6.48455575,   58.36100178,
	         25.93822301,   64.84555753,   94.02605842,   71.33011328,
	         84.29922479,   71.33011328,   71.33011328,   55.1187239 ,
	         25.93822301,   19.45366726,   74.57239116,   71.33011328,
	         35.66505664,   74.57239116,   35.66505664,  100.51061417,
	         77.81466904,    3.24227788,   35.66505664,   51.87644602,
	         22.69594514,   64.84555753,   61.60327965,   68.08783541,
	         71.33011328,   94.02605842,   16.21138938,   38.90733452,
	         16.21138938,   22.69594514,   45.39189027,   45.39189027,
	         29.18050089,   12.96911151,   29.18050089,    6.48455575,
	          9.72683363,   38.90733452,    6.48455575,   45.39189027,
	          9.72683363,   25.93822301,   29.18050089,   38.90733452,
	          6.48455575,   35.66505664,   19.45366726,   32.42277877,
	         25.93822301,   19.45366726,   32.42277877,   29.18050089,
	         25.93822301,   19.45366726,   38.90733452,   19.45366726,
	         25.93822301,   19.45366726,   45.39189027,   38.90733452,
	         16.21138938,   16.21138938,   35.66505664,    6.48455575,
	         38.90733452,   42.1496124 ,   25.93822301,   42.1496124 ,
	         45.39189027,   12.96911151,   48.63416815,   42.1496124 ,
	         51.87644602,   38.90733452,    9.72683363,   45.39189027,
	          6.48455575,   25.93822301,   16.21138938,   22.69594514,
	         29.18050089,    6.48455575,    3.24227788,    9.72683363,
	          3.24227788,    3.24227788,   22.69594514,   32.42277877,
	         42.1496124 ,   12.96911151,    6.48455575,   22.69594514,
	         29.18050089,    6.48455575,    6.48455575,    6.48455575,
	         29.18050089,   19.45366726,    9.72683363,   32.42277877,
	         25.93822301,   16.21138938,   38.90733452,   38.90733452,
	         29.18050089,    6.48455575,   12.96911151,    6.48455575,
	         29.18050089,   25.93822301,    3.24227788,    3.24227788,
	         16.21138938,   22.69594514,   42.1496124 ,   25.93822301,
	         35.66505664,   12.96911151,   25.93822301,   25.93822301,
	         22.69594514,   25.93822301,   19.45366726,   42.1496124 ,
	          9.72683363,   29.18050089
		};
		
		expected_log = new double[]{
			4.39515196,  4.12071511,  4.26731858,  4.06664789,  4.12071511,
	        3.5741714 ,  4.1720084 ,  4.67278369,  3.5741714 ,  4.31177035,
	        4.06664789,  4.57747351,  4.1720084 ,  2.78571404,  2.27488842,
	        1.86942331,  3.5741714 ,  4.39515196,  2.56257049,  4.12071511,
	        3.74122549,  4.26731858,  1.86942331,  4.06664789,  3.25571767,
	        4.1720084 ,  4.54357196,  4.26731858,  4.43437267,  4.26731858,
	        4.26731858,  4.00948948,  3.25571767,  2.9680356 ,  4.31177035,
	        4.26731858,  3.5741714 ,  4.31177035,  3.5741714 ,  4.61026334,
	        4.35432996,  1.17627613,  3.5741714 ,  3.94886485,  3.12218628,
	        4.1720084 ,  4.12071511,  4.22079857,  4.26731858,  4.54357196,
	        2.78571404,  3.66118278,  2.78571404,  3.12218628,  3.81533346,
	        3.81533346,  3.37350071,  2.56257049,  3.37350071,  1.86942331,
	        2.27488842,  3.66118278,  1.86942331,  3.81533346,  2.27488842,
	        3.25571767,  3.37350071,  3.66118278,  1.86942331,  3.5741714 ,
	        2.9680356 ,  3.47886122,  3.25571767,  2.9680356 ,  3.47886122,
	        3.37350071,  3.25571767,  2.9680356 ,  3.66118278,  2.9680356 ,
	        3.25571767,  2.9680356 ,  3.81533346,  3.66118278,  2.78571404,
	        2.78571404,  3.5741714 ,  1.86942331,  3.66118278,  3.74122549,
	        3.25571767,  3.74122549,  3.81533346,  2.56257049,  3.88432633,
	        3.74122549,  3.94886485,  3.66118278,  2.27488842,  3.81533346,
	        1.86942331,  3.25571767,  2.78571404,  3.12218628,  3.37350071,
	        1.86942331,  1.17627613,  2.27488842,  1.17627613,  1.17627613,
	        3.12218628,  3.47886122,  3.74122549,  2.56257049,  1.86942331,
	        3.12218628,  3.37350071,  1.86942331,  1.86942331,  1.86942331,
	        3.37350071,  2.9680356 ,  2.27488842,  3.47886122,  3.25571767,
	        2.78571404,  3.66118278,  3.66118278,  3.37350071,  1.86942331,
	        2.56257049,  1.86942331,  3.37350071,  3.25571767,  1.17627613,
	        1.17627613,  2.78571404,  3.12218628,  3.74122549,  3.25571767,
	        3.5741714 ,  2.56257049,  3.25571767,  3.25571767,  3.12218628,
	        3.25571767,  2.9680356 ,  3.74122549,  2.27488842,  3.37350071
		};
		
		assertTrue(VecUtils.equalsWithTolerance(exp, expected_exp, 1e-8));
		assertTrue(VecUtils.equalsWithTolerance(log, expected_log, 1e-8));
		
		
		
		/*
		 * EPANECHNIKOV
		 */
		exp = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_EPANECHNIKOV, at, rt, false);
		log = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_EPANECHNIKOV, at, rt, true);
		expected_exp = new double[]{
			136.56474416,  107.77331662,  118.27829694,  112.05312341,
	        114.77663683,   45.91065473,   69.25505544,  161.07636491,
	         49.80138818,  126.44883719,   90.26501608,  124.50347046,
	        110.88590338,   28.79142754,   14.00664043,   14.39571377,
	         38.90733452,  136.95381751,   20.23181395,   96.10111626,
	         53.69212164,  109.71868334,   11.28312701,   57.58285509,
	         28.79142754,   89.87594274,  121.77995704,  135.39752412,
	        128.39420391,  119.83459032,  122.94717708,   72.3676422 ,
	         31.51494096,   27.23513416,  126.44883719,   91.04316277,
	         48.63416815,  126.44883719,   59.13914847,  152.12767797,
	        109.32961   ,    9.72683363,   60.69544185,   52.91397495,
	         25.67884078,  107.77331662,   84.81798925,  113.22034345,
	        108.94053665,  157.5747048 ,   26.06791413,   48.2450948 ,
	         34.23845438,   36.1838211 ,   48.63416815,   55.24841502,
	         29.18050089,   24.1225474 ,   49.41231484,   13.61756708,
	         15.95200715,   49.41231484,   10.11590697,   48.63416815,
	         14.39571377,   44.35436135,   39.29640786,   50.57953487,
	         16.73015384,   64.5861753 ,   32.29308765,   36.96196779,
	         30.34772092,   31.51494096,   45.13250804,   52.5249016 ,
	         34.62752772,   28.4023542 ,   51.74675491,   22.95532737,
	         47.07787477,   35.40567441,   66.53154203,   42.01992128,
	         22.17718068,   19.45366726,   57.97192843,   16.73015384,
	         53.69212164,   61.47358854,   31.90401431,   53.69212164,
	         67.69876206,   27.23513416,   77.42559569,   59.52822181,
	         78.20374238,   47.07787477,   17.50830053,   82.87262252,
	         12.45034705,   43.96528801,   21.78810733,   36.57289445,
	         38.51826117,   16.73015384,    9.72683363,   19.06459391,
	          9.72683363,    9.72683363,   28.79142754,   38.90733452,
	         48.2450948 ,   29.18050089,   10.11590697,   34.23845438,
	         42.79806797,   12.83942039,   12.83942039,   12.0612737 ,
	         50.57953487,   28.79142754,   19.8427406 ,   48.2450948 ,
	         40.85270124,   22.17718068,   48.63416815,   52.5249016 ,
	         43.57621466,   14.78478712,   19.45366726,   12.83942039,
	         33.84938103,   30.34772092,    9.72683363,    9.72683363,
	         26.06791413,   35.01660107,   50.96860822,   39.29640786,
	         52.13582826,   22.95532737,   43.96528801,   41.63084793,
	         35.79474776,   40.4636279 ,   29.18050089,   47.46694811,
	         23.34440071,   44.7434347
		};
		
		expected_log = new double[]{
			4.91679882,  4.6800301 ,  4.7730403 ,  4.71897308,  4.74298795,
	        3.82669722,  4.23779615,  5.08187857,  3.90804286,  4.83983778,
	        4.50274997,  4.82433359,  4.70850178,  3.36007769,  2.63953153,
	        2.66693051,  3.66118278,  4.91964377,  3.00725631,  4.56540093,
	        3.98326628,  4.69791967,  2.42330842,  4.05322487,  3.36007769,
	        4.49843031,  4.80221579,  4.90821507,  4.85510525,  4.78611238,
	        4.81175481,  4.28175927,  3.45046175,  3.30450784,  4.83983778,
	        4.51133371,  3.88432633,  4.83983778,  4.07989312,  5.02472015,
	        4.69436726,  2.27488842,  4.1058686 ,  3.96866748,  3.24566734,
	        4.6800301 ,  4.44050766,  4.72933586,  4.6908022 ,  5.05989966,
	        3.26070521,  3.87629416,  3.53334941,  3.58861209,  3.88432633,
	        4.01183965,  3.37350071,  3.18314698,  3.90019968,  2.61136066,
	        2.76958466,  3.90019968,  2.31410913,  3.88432633,  2.66693051,
	        3.79221104,  3.67113311,  3.92354705,  2.81721271,  4.16800038,
	        3.4748532 ,  3.60988949,  3.41272142,  3.45046175,  3.80960279,
	        3.96128737,  3.54464896,  3.34647204,  3.94636172,  3.13355004,
	        3.85180314,  3.5668721 ,  4.19767615,  3.73814382,  3.09906386,
	        2.9680356 ,  4.0599589 ,  2.81721271,  3.98326628,  4.11860763,
	        3.46273184,  3.98326628,  4.21506789,  3.30450784,  4.34931742,
	        4.08645052,  4.3593175 ,  3.85180314,  2.86267508,  4.41730476,
	        2.5217485 ,  3.78340041,  3.08136429,  3.59930738,  3.65113245,
	        2.81721271,  2.27488842,  2.94783289,  2.27488842,  2.27488842,
	        3.36007769,  3.66118278,  3.87629416,  3.37350071,  2.31410913,
	        3.53334941,  3.75649296,  2.55252016,  2.55252016,  2.4899998 ,
	        3.92354705,  3.36007769,  2.98783823,  3.87629416,  3.70997295,
	        3.09906386,  3.88432633,  3.96128737,  3.77451147,  2.69359875,
	        2.9680356 ,  2.55252016,  3.52192071,  3.41272142,  2.27488842,
	        2.27488842,  3.26070521,  3.55582227,  3.93120992,  3.67113311,
	        3.95385239,  3.13355004,  3.78340041,  3.72884143,  3.57780117,
	        3.70040349,  3.37350071,  3.86003364,  3.15035716,  3.80094472
		};

		assertTrue(VecUtils.equalsWithTolerance(exp, expected_exp, 1e-8));
		assertTrue(VecUtils.equalsWithTolerance(log, expected_log, 1e-8));
		
		
		/*
		 * EXPONENTIAL
		 */
		exp = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_EXPONENTIAL, at, rt, false);
		log = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_EXPONENTIAL, at, rt, true);
		expected_exp = new double[]{
			2.81678649,  2.44420675,  2.46085198,  2.32334735,  2.66174388,
	        1.75833401,  2.26258436,  2.90604716,  1.66101735,  2.65197964,
	        2.21241444,  2.61487079,  2.36662588,  1.3429302 ,  1.08582846,
	        0.87810901,  1.74938498,  2.80594815,  1.43966768,  2.33798563,
	        2.09649275,  2.44929755,  1.52013833,  2.24654141,  1.97405298,
	        2.3113064 ,  2.66193891,  2.73534441,  2.69575639,  2.49807964,
	        2.5396906 ,  2.18998074,  1.57014777,  1.29623897,  2.65197964,
	        2.36627864,  1.93596341,  2.65197964,  1.76027591,  2.87279821,
	        2.63973029,  0.80072927,  1.86171852,  2.17588605,  1.74625427,
	        2.36335638,  2.27009911,  2.39545832,  2.38680074,  2.84340518,
	        1.47476553,  2.20254506,  1.80134266,  1.64905292,  2.37559262,
	        2.3070159 ,  2.1904533 ,  0.84587259,  2.13461142,  1.43116099,
	        0.76391881,  2.25300808,  1.36079879,  2.61769873,  1.43585157,
	        1.84406234,  2.13951774,  1.98781592,  1.67412091,  1.86701684,
	        2.16352523,  2.00735168,  2.20720179,  2.29847941,  2.13683403,
	        2.06054352,  1.89680547,  2.22597632,  2.60765944,  1.31116214,
	        1.63714073,  1.46722224,  2.04579549,  2.32608004,  1.78748257,
	        1.86802593,  2.10754762,  1.70910096,  2.04791726,  1.91072784,
	        1.92581394,  2.58256346,  2.1324481 ,  0.88450342,  2.25258946,
	        2.15077007,  2.33978932,  2.32637351,  0.77329659,  2.31090614,
	        1.12911703,  2.11289612,  1.45396034,  2.02929272,  1.81955277,
	        0.73616377,  0.91562102,  0.97918172,  1.36387393,  0.82200445,
	        2.21816716,  2.26983463,  2.0555581 ,  1.70486348,  1.42975657,
	        1.96151767,  2.21193959,  0.46586307,  0.46478071,  1.59243667,
	        1.70298342,  1.7806412 ,  0.63135066,  2.52095548,  1.81997945,
	        1.210823  ,  2.6026903 ,  2.58550811,  2.02807087,  1.21502588,
	        1.03827884,  0.47923644,  1.94235239,  2.39078077,  1.42544878,
	        0.78684905,  1.50742558,  2.15781815,  2.5164454 ,  1.93006564,
	        1.81774451,  1.62710122,  2.11289612,  1.58247195,  1.52460124,
	        1.92625489,  2.12969908,  2.39554574,  1.60979498,  2.25995104
		};
		
		expected_log = new double[]{
			1.03559669,  0.89372063,  0.90050762,  0.84300897,  0.9789815 ,
	        0.56436677,  0.81650768,  1.06679379,  0.50743028,  0.9753064 ,
	        0.79408443,  0.96121468,  0.86146526,  0.29485394,  0.08234326,
	       -0.12998453,  0.55926429,  1.0317415 ,  0.36441231,  0.84928972,
	        0.74026583,  0.89580127,  0.41880134,  0.80939188,  0.68008878,
	        0.8378129 ,  0.97905477,  1.00625736,  0.99167883,  0.91552229,
	        0.93204226,  0.78389275,  0.45116974,  0.25946697,  0.9753064 ,
	        0.86131853,  0.66060509,  0.9753064 ,  0.56547056,  1.05528654,
	        0.97067675, -0.22223238,  0.6215    ,  0.77743596,  0.55747307,
	        0.8600828 ,  0.81982349,  0.87357458,  0.86995387,  1.04500234,
	        0.38849901,  0.78961354,  0.58853231,  0.50020114,  0.86524693,
	        0.83595487,  0.78410851, -0.16738653,  0.75828462,  0.358486  ,
	       -0.26929377,  0.81226625,  0.30807188,  0.96229558,  0.3617581 ,
	        0.61197093,  0.76058045,  0.68703651,  0.5152882 ,  0.62434188,
	        0.77173894,  0.69681628,  0.79172556,  0.83224778,  0.75932531,
	        0.72296979,  0.64017114,  0.80019562,  0.95845305,  0.27091387,
	        0.49295126,  0.38337098,  0.7157867 ,  0.84418446,  0.58080824,
	        0.62488222,  0.74552501,  0.53596748,  0.7168233 ,  0.64748424,
	        0.6553487 ,  0.94878249,  0.75727066, -0.12272889,  0.81208043,
	        0.76582595,  0.85006089,  0.84431062, -0.25709261,  0.83763972,
	        0.12143594,  0.74805957,  0.3742911 ,  0.70768732,  0.59859074,
	       -0.30630268, -0.08815273, -0.02103803,  0.31032913, -0.19600947,
	        0.79668125,  0.81970698,  0.72054739,  0.53348504,  0.3575042 ,
	        0.6737185 ,  0.79386977, -0.76386354, -0.76618957,  0.46526534,
	        0.53238167,  0.57697352, -0.45989384,  0.92463799,  0.59882521,
	        0.1913003 ,  0.95654564,  0.94992205,  0.70708503,  0.19476537,
	        0.03756438, -0.7355612 ,  0.66389981,  0.87162   ,  0.3544867 ,
	       -0.23971885,  0.41040328,  0.76909759,  0.92284735,  0.65755401,
	        0.59759645,  0.48680004,  0.74805957,  0.45898815,  0.4217329 ,
	        0.65557765,  0.75598069,  0.87361107,  0.47610683,  0.81534315
		};

		assertTrue(VecUtils.equalsWithTolerance(exp, expected_exp, 1e-8));
		assertTrue(VecUtils.equalsWithTolerance(log, expected_log, 1e-8));
		
		
		
		/*
		 * LINEAR
		 */
		exp = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_LINEAR, at, rt, false);
		log = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_LINEAR, at, rt, true);
		expected_exp = new double[]{
			160.9282694 ,  127.53018651,  130.14235947,  127.06395876,
	        132.09672819,   50.65924756,   75.57647598,  179.90963654,
	         58.10252844,  154.52998575,   98.87756863,  131.51181617,
	        129.0150588 ,   36.08103914,   20.05828275,   20.73257963,
	         43.47854467,  160.11680757,   26.32721888,  107.17710239,
	         58.7694493 ,  120.17478496,   17.56479497,   63.23096165,
	         34.44799898,  101.58528645,  131.42827689,  156.00513214,
	        146.21446149,  134.0154519 ,  142.01816604,   77.61149955,
	         36.76669062,   32.87159631,  154.52998575,   96.51714225,
	         54.07440652,  154.52998575,   70.24275874,  173.00716974,
	        124.08736307,   16.21138938,   70.20707524,   58.95239275,
	         31.16071744,  121.51159135,   93.91208357,  128.14529958,
	        120.81621787,  174.91680235,   32.73452138,   53.63978727,
	         41.82357635,   43.89184638,   53.92778303,   60.38366728,
	         35.08006041,   32.84501217,   57.6022944 ,   19.86549055,
	         22.08598522,   54.33870995,   16.53892596,   58.11028608,
	         20.37844064,   52.92242602,   46.16945232,   56.92813616,
	         23.84451782,   74.68419345,   39.63789851,   41.95799374,
	         35.34634228,   38.74351531,   52.36152552,   63.04288352,
	         40.34582251,   34.0039662 ,   59.29449178,   28.4406359 ,
	         58.0248369 ,   44.60769876,   75.06894691,   46.66720883,
	         29.77086576,   25.10401887,   64.17972879,   23.84451782,
	         64.25783636,   69.54657107,   38.15913986,   62.91758961,
	         77.74249696,   36.01279683,   87.08271188,   70.25392351,
	         91.88226395,   53.6381145 ,   23.51959171,   95.98586896,
	         18.66695873,   57.31741601,   27.30430924,   44.31795605,
	         44.22810104,   23.84451782,   16.21138938,   25.92314559,
	         16.21138938,   16.21138938,   35.49515485,   43.55055473,
	         54.76906451,   36.93561646,   16.53892596,   39.87542221,
	         51.72322503,   19.05452461,   19.05452461,   18.29001715,
	         58.62218765,   35.26711472,   26.68765305,   56.23803193,
	         46.55365872,   27.9276858 ,   56.76518959,   61.99328508,
	         52.76666665,   21.19119874,   26.20168155,   19.05452461,
	         43.21052816,   35.70999588,   16.21138938,   16.21138938,
	         32.66915064,   44.19686106,   60.37715009,   46.61929426,
	         58.6886308 ,   30.00580075,   57.31741601,   48.89422033,
	         42.90669119,   46.59416438,   35.71995611,   53.43249648,
	         30.96540812,   51.08482935
		};
		
		expected_log = new double[]{
			5.08095873,  4.84835309,  4.86862892,  4.84469057,  4.88353444,
	        3.92512179,  4.32514507,  5.19245471,  4.06220918,  5.04038816,
	        4.5938824 ,  4.8790967 ,  4.85992913,  3.5857675 ,  2.99864217,
	        3.03170636,  3.77226759,  5.0759036 ,  3.27060334,  4.67448263,
	        4.07362215,  4.78894722,  2.86589661,  4.14679408,  3.53945091,
	        4.62089871,  4.87846128,  5.04988891,  4.98507446,  4.89795511,
	        4.95595498,  4.35171561,  3.60459229,  3.49260895,  5.04038816,
	        4.56972063,  3.990361  ,  5.04038816,  4.25195722,  5.15333304,
	        4.82098586,  2.78571404,  4.25144909,  4.07673022,  3.43915825,
	        4.80000966,  4.54235906,  4.85316477,  4.79427053,  5.16431045,
	        3.48843022,  3.98229109,  3.73346021,  3.78172857,  3.9876458 ,
	        4.10071866,  3.55763289,  3.4917999 ,  4.0535624 ,  2.98898408,
	        3.09494325,  3.99523686,  2.80571675,  4.06234269,  3.01447751,
	        3.96882718,  3.83231837,  4.0417897 ,  3.17155433,  4.31326847,
	        3.67978569,  3.73666897,  3.56519492,  3.65696339,  3.95817208,
	        4.14381519,  3.69748786,  3.52647717,  4.08251641,  3.34781896,
	        4.06087114,  3.79790646,  4.31840698,  3.84304175,  3.39353026,
	        3.22302795,  4.16168741,  3.17155433,  4.16290368,  4.24199662,
	        3.64176531,  4.14182577,  4.35340204,  3.58387434,  4.46685838,
	        4.25211616,  4.52050802,  3.98225991,  3.15783376,  4.56420098,
	        2.92675505,  4.04860452,  3.30704454,  3.79138992,  3.78936036,
	        3.17155433,  2.78571404,  3.25513622,  2.78571404,  2.78571404,
	        3.5693962 ,  3.77392244,  4.00312552,  3.6091763 ,  2.80571675,
	        3.68576015,  3.94590691,  2.94730459,  2.94730459,  2.9063554 ,
	        4.07111325,  3.56295094,  3.28420103,  4.02959325,  3.8406056 ,
	        3.32961852,  4.03892328,  4.12702607,  3.96587968,  3.05358594,
	        3.26582359,  2.94730459,  3.76608417,  3.57543065,  2.78571404,
	        2.78571404,  3.48643123,  3.78865377,  4.10061072,  3.8420145 ,
	        4.07224602,  3.40139072,  4.04860452,  3.8896592 ,  3.75902779,
	        3.84147531,  3.57570953,  3.97841911,  3.43287071,  3.93348757
		};
		
		assertTrue(VecUtils.equalsWithTolerance(exp, expected_exp, 1e-8));
		assertTrue(VecUtils.equalsWithTolerance(log, expected_log, 1e-8));
		assertTrue(new VecSeries(log, Inequality.NOT_EQUAL_TO, Double.NaN).all());
		
		/*
		 * COSINE
		 */
		exp = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_COSINE, at, rt, false);
		log = k.kernelDensity(IRIS.getData(), h, PartialKernelDensity.LOG_COSINE, at, rt, true);
		
		// all should be nan...
		assertTrue(new VecSeries(exp, Inequality.EQUAL_TO, Double.NaN).all());
		assertTrue(new VecSeries(log, Inequality.EQUAL_TO, Double.NaN).all());
	}
	
	
	
	// ================== constructor tests
	@Test
	public void testConst1() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		Loggable log = null;
		
		// test kd constructors
		KDTree kd = new KDTree(A);
		kd = new KDTree(A, 5);
		kd = new KDTree(A, 5, Distance.EUCLIDEAN);
		assertTrue(kd.getLeafSize() == 5);
		kd = new KDTree(A, Distance.EUCLIDEAN);
		kd = new KDTree(A, log);
		assertTrue(kd.logger == null);
		kd = new KDTree(A, 5, Distance.EUCLIDEAN, null);
		
		
		BallTree ball = new BallTree(A);
		ball = new BallTree(A, 5);
		ball = new BallTree(A, 5, Distance.EUCLIDEAN);
		assertTrue(5 == ball.getLeafSize());
		ball = new BallTree(A, Distance.EUCLIDEAN);
		ball = new BallTree(A, log);
		assertTrue(ball.logger == null);
		ball = new BallTree(A, 5, Distance.EUCLIDEAN, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstIAE1() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		new KDTree(A, 0);
	}
	
	// Create anonymous DistanceMetric class to test
	@Test
	public void testConst2() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A, new DistanceMetric() {
			private static final long serialVersionUID = 6792348831585297421L;

			@Override public double getDistance(final double[] a, final double[] b) { return 0.0; }
			@Override public double getP() { return 0.0; }
			@Override public double getPartialDistance(final double[] a, final double[] b) { return getDistance(a, b); }
			@Override public double partialDistanceToDistance(double d) { return d; }
			@Override public double distanceToPartialDistance(double d) { return d; }
			@Override public String getName() { return "Test anonymous DistanceMetric"; }
		});
		
		assertTrue(kd.getMetric().equals(Distance.EUCLIDEAN));
	}
	
	private static void passByRef(MutableDouble md, double x) {
		md.value = x;
	}
	
	@Test
	public void testMutableDouble() {
		MutableDouble md = new MutableDouble(145d);
		passByRef(md, 15d);
		assertTrue(md.value == 15d);
		assertTrue(md.compareTo(14d) == 1);
		assertTrue(new MutableDouble().value == 0d);
	}
	
	@Test
	public void testNodeDataContainerClass() {
		// Test def constructor
		NodeData node = new NodeData();
		assertTrue(node.idx_start == 0);
		assertTrue(node.idx_end == 0);
		assertTrue(!node.is_leaf);
		assertTrue(node.radius == 0.0);
		
		// Test arg constructor
		node = new NodeData(1,2,true,5.9);
		assertTrue(node.idx_start == 1);
		assertTrue(node.idx_end == 2);
		assertTrue(node.is_leaf);
		assertTrue(node.radius == 5.9);
		
		// Test immutability
		NodeData node2 = node.copy();
		node2.idx_start = 15;
		node2.idx_end = 67;
		node2.is_leaf = false;
		node2.radius = 5.6;
		assertTrue(node.start() == 1);
		assertTrue(node.end() == 2);
		assertTrue(node.isLeaf());
		assertTrue(node.radius() == 5.9);
		
		// ensure won't throw exception
		node.toString();
	}
	
	@Test
	public void testGetterRefMutability() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		
		double[][] data = kd.getData();
		double[][] dataRef = kd.getDataRef();
		dataRef[0][0] = 150d;
		assertFalse(MatUtils.equalsExactly(kd.getDataRef(), data));
		
		double[][][] bounds = kd.getNodeBounds();
		double[][][] boundsRef = kd.getNodeBoundsRef();
		boundsRef[0][0][0] = 150;
		assertFalse(MatUtils.equalsExactly(kd.getNodeBoundsRef()[0], bounds[0]));
		
		int[] idcs = kd.getIndexArray();
		int[] idcsRef = kd.getIndexArrayRef();
		idcsRef[0] = 150;
		assertFalse(VecUtils.equalsExactly(kd.getIndexArrayRef(), idcs));
		
		NodeData[] nodes = kd.getNodeData();
		NodeData[] nodeRef=kd.getNodeDataRef();
		nodeRef[0].idx_end = 150;
		assertFalse(kd.getNodeDataRef()[0].idx_end == nodes[0].idx_end);
	}
	
	@Test
	public void testInstanceMethod() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		
		double[] b = new double[]{0,1,2};
		double[] c = new double[]{3,4,5};
		assertTrue(kd.dist(b,c) == Distance.EUCLIDEAN.getDistance(b, c));
		assertTrue(kd.rDist(b,c) == Distance.EUCLIDEAN.getPartialDistance(b, c));
		assertTrue(kd.rDistToDist(kd.rDist(b, c)) == Distance.EUCLIDEAN.partialDistanceToDistance(kd.rDist(b, c)));
		assertTrue(kd.getNumCalls() == 4);
		kd.resetNumCalls();
		
		assertTrue(kd.getNumCalls() == 0);
	}
	
	@Test
	public void testNodeFind() {
		Array2DRowRealMatrix A = new Array2DRowRealMatrix(a);
		KDTree kd = new KDTree(A);
		final int findNode = KDTree.findNodeSplitDim(a, kd.idx_array);
		assertTrue(findNode == 2);
	}
	
	@Test
	public void testSwap() {
		int[] ex = new int[]{0,1,2};
		KDTree.swap(ex, 0, 1);
		assertTrue(VecUtils.equalsExactly(ex, new int[]{1,0,2}));
	}
	
	@Test(expected=IllegalStateException.class)
	public void testNodeHeap1() {
		NodeHeap nh1 = new NodeHeap(0);
		assertTrue(nh1.data.length == 1); // picks max (size, 1)
		nh1 = new NodeHeap(2);
		assertTrue(nh1.data.length == 2);
		assertTrue(nh1.n == 0);
		nh1.clear();
		assertTrue(nh1.n == 0);
		assertTrue(null == nh1.peek());
		nh1.pop(); // throws the exception on empty heap
	}
	
	@Test
	public void testNodeHeapPushesPops() {
		NodeHeap heap = new NodeHeap(3);
		
		NodeHeapData h = new NodeHeapData(1.0,  0, 0);
		heap.push(new NodeHeapData(12.0, 1, 2));
		heap.push(new NodeHeapData(9.0,  4, 5));
		heap.push(new NodeHeapData(11.0, 9,-1));
		heap.push(h);
		
		assertTrue(heap.data.length == 8);
		assertTrue(heap.data[0].val == 1.0);
		assertTrue(heap.data[1].val == 9.0);
		assertTrue(heap.data[2].val ==11.0);
		assertTrue(heap.data[3].val ==12.0);
		
		assertTrue(heap.data[0].i1 == 0);
		assertTrue(heap.data[1].i1 == 4);
		assertTrue(heap.data[2].i1 == 9);
		assertTrue(heap.data[3].i1 == 1);
		
		assertTrue(heap.data[0].i2 == 0);
		assertTrue(heap.data[1].i2 == 5);
		assertTrue(heap.data[2].i2 ==-1);
		assertTrue(heap.data[3].i2 == 2);
		
		assertTrue(heap.data[0].equals(new NodeHeapData(1.0,0,0)));
		assertTrue(heap.data[0].equals(h));
		assertFalse(heap.data[0].equals(new Integer(1)));
		assertTrue(heap.n == 4);
		
		assertTrue(heap.pop().equals(h));
		assertTrue(heap.data[0].val == 9.0);
		assertTrue(heap.data[1].val ==12.0);
		assertTrue(heap.data[2].val ==11.0);
		assertTrue(null == heap.data[3]);
		
		// Ensure no NPE
		heap.toString();
	}
	
	@Test
	public void testDualSwap() {
		double[] a = new double[]{0,1,2};
		int[] b = new int[]{3,4,5};
		Heap.dualSwap(a, b, 0, 1);
		assertTrue(VecUtils.equalsExactly(a, new double[]{1,0,2}));
		assertTrue(VecUtils.equalsExactly(b, new int[]{4,3,5}));
	}
	
	@Test
	public void testBigKD() {
		Array2DRowRealMatrix x = new Array2DRowRealMatrix(IRIS.getData(),false);
		KDTree kd = new KDTree(x);
		assertTrue(VecUtils.equalsExactly(kd.idx_array, new int[]{
			0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,
	        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
	        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
	        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  53,  57,
	        59,  60,  62,  64,  69,  71,  79,  80,  81,  82,  89,  92,  93,
	        98,  99,  88,  67,  61,  94,  95,  96,  74,  97,  90,  87,  65,
	        75, 106,  86,  68,  54,  55,  73,  91,  56,  63,  78,  51,  58,
	        66,  50,  84,  85, 138,  76,  70,  52, 121, 123, 126, 127,  72,
	       146,  77, 113, 119, 149, 109, 110, 111, 112, 100, 114, 115, 116,
	       117, 118, 101, 120, 102, 122, 103, 124, 125, 104, 105, 128, 129,
	       130, 131, 132, 133, 134, 135, 136, 137,  83, 139, 140, 141, 142,
	       143, 144, 145, 107, 147, 148, 108
		}));
	}
	
	@Test
	public void testQuerySmall() {
		KDTree kd = new KDTree(new Array2DRowRealMatrix(a));
		assertTrue(VecUtils.equalsExactly(kd.idx_array, new int[]{0,1,2}));
		assertTrue(kd.node_bounds.length == 2);
		assertTrue(kd.node_bounds[0].length == 1);
		assertTrue(kd.node_bounds[1].length == 1);
		assertTrue(VecUtils.equalsExactly(kd.node_bounds[0][0], new double[]{0.0,0.0,0.0,2.0}));
		assertTrue(VecUtils.equalsExactly(kd.node_bounds[1][0], new double[]{5.0,6.0,7.0,4.0}));
	
		
		double[][] expectedDists = new double[][]{ new double[]{0.0}, new double[]{0.0} };
		int[][] expectedIndices  = new int[][]{ new int[]{0}, new int[]{1} };
		
		Neighborhood neighb;
		boolean[] trueFalse = new boolean[]{true, false};
		for(boolean dualTree: trueFalse) {
			for(boolean sort: trueFalse) {
				neighb= new Neighborhood(
					kd.query(new double[][]{
						new double[]{0,1,0,2},
						new double[]{0,0,1,2}
					}, 1, dualTree, sort));
				
				assertTrue(MatUtils.equalsExactly(expectedDists, neighb.getDistances()));
				assertTrue(MatUtils.equalsExactly(expectedIndices, neighb.getIndices()));
			}
		}
	}
	
	@Test
	public void testSimultaneousSort() {
		double[] dists = new double[]{
			3.69675274,  2.89351805,  1.79065633,  
			0.44375205,  7.77409946,  7.08011014,
			8.41547227,  5.57512117,  8.85578907,
			2.60367035 };
		
		int[] indices = new int[]{
			4, 1, 0, 7, 6, 5, 8, 2, 3, 9
		};
		
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.44375205,  1.79065633,  2.60367035,
			2.89351805,  3.69675274,  5.57512117,
			7.08011014,  7.77409946,  8.41547227,
			8.85578907
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			7, 0, 9, 1, 4, 2, 5, 6, 8, 3
		}));
		
		
		
		
		dists = new double[]{0.7,0.1};
		indices = new int[]{2,1};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.1,0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			1,2
		}));
		
		
		
		
		dists = new double[]{0.7};
		indices = new int[]{2};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			2
		}));
		
		
		
		dists = new double[]{0.7,0.1,0.3};
		indices = new int[]{2,1,3};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.1,0.3,0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			1,3,2
		}));
		
		
		
		dists = new double[]{0.3,0.7,0.1};
		indices = new int[]{2,1,3};
		NeighborsHeap.simultaneous_sort(dists, indices, dists.length);
		assertTrue(VecUtils.equalsExactly(dists, new double[]{
			0.1,0.3,0.7
		}));
		
		assertTrue(VecUtils.equalsExactly(indices, new int[]{
			3,2,1
		}));
	}
	
	@Test
	public void testNeighborsHeap() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195, -0.48111094},
			new double[]{-0.95392127,  0.72765662,  0.46466226},
			new double[]{ 0.57011545, -1.53581033,  0.52009414}
		};
		
		final int k = 1;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(true));
		
		double[][] dists = neighb.getDistances();
		int[][] inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.48111094},
			new double[]{-0.95392127},
			new double[]{-1.53581033}
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{2},
			new int[]{0},
			new int[]{1}
		}));
	}
	
	@Test
	public void testNeighborHeapOrderInPlace() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195, -0.48111094,  0.0002354, 1.12345},
			new double[]{-0.95392127,  0.72765662,  0.46466226, -0.9128421, 5.12345},
			new double[]{ 0.57011545, -1.53581033,  0.52009414,  0.1958271, -4.3918}
		};
		
		final int k = 3;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(true));
		
		double[][] dists = neighb.getDistances();
		int[][] inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.48111094, -0.26063195, 0.0002354},
			new double[]{-0.95392127, -0.9128421, 0.46466226},
			new double[]{-4.3918,     -1.53581033, 0.1958271}
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{2,1,3},
			new int[]{0,3,2},
			new int[]{4,1,3}
		}));
	}
	
	@Test
	public void testNeighborHeapTwoAndLessLen() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195},
			new double[]{-0.95392127,  0.72765662},
			new double[]{ 0.57011545, -1.53581033}
		};
		
		
		
		int k = 1;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(true));
		
		double[][] dists = neighb.getDistances();
		int[][] inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.26063195},
			new double[]{-0.95392127},
			new double[]{-1.53581033}
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{1},
			new int[]{0},
			new int[]{1}
		}));
		
		
		
		k = 2;
		heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		neighb = new Neighborhood(heap.getArrays(true));
		
		dists = neighb.getDistances();
		inds = neighb.getIndices();
		
		assertTrue(MatUtils.equalsExactly(dists, new double[][]{
			new double[]{-0.26063195, 0.15464338 },
			new double[]{-0.95392127, 0.72765662 },
			new double[]{-1.53581033, 0.57011545 }
		}));
		
		assertTrue(MatUtils.equalsExactly(inds, new int[][]{
			new int[]{1,0},
			new int[]{0,1},
			new int[]{1,0}
		}));
	}
	
	@Test
	public void testNeighborHeapNoSortAndLargest() {
		double[][] X = new double[][]{
			new double[]{ 0.15464338, -0.26063195, -0.48111094,  0.0002354, 1.12345},
			new double[]{-0.95392127,  0.72765662,  0.46466226, -0.9128421, 5.12345},
			new double[]{ 0.57011545, -1.53581033,  0.52009414,  0.1958271, -4.3918}
		};
		
		final int k = 3;
		NeighborsHeap heap = new NeighborsHeap(X.length, k);
		for(int i = 0; i < X.length; i++)
			for(int j = 0; j < X[0].length; j++)
				heap.push(i, X[i][j], j);
		Neighborhood neighb = new Neighborhood(heap.getArrays(false));
		
		double[][] dists = neighb.getDistances();
		for(int row = 0; row < dists.length; row++)
			assertTrue(heap.largest(row) == VecUtils.max(dists[row]));
	}
	
	@Test
	public void testDistToRDist() {
		double[]a = new double[]{5,0,0};
		double[]b = new double[]{0,0,1};
		KDTree kd = new KDTree(IRIS);
		assertTrue(kd.dist(a, b) == 5.0990195135927845);
		assertTrue(kd.rDistToDist(25.999999999999996) == kd.dist(a, b));
		assertTrue(Precision.equals(kd.rDist(a, b), 25.999999999999996, 1e-8));
		assertTrue(Precision.equals(kd.rDistToDist(kd.rDist(a, b)), kd.dist(a, b), 1e-8));
	}
	
	@Test
	public void testMinRDistDual() {
		Array2DRowRealMatrix X1 = IRIS;
		
		double[][] query = new double[10][];
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		Array2DRowRealMatrix X2 = new Array2DRowRealMatrix(query, false);
		
		NearestNeighborHeapSearch tree1 = new KDTree(X1);
		NearestNeighborHeapSearch tree2 = new KDTree(X2);
		
		double dist = tree1.minRDistDual(tree1, 0, tree2, 0);
		assertTrue(0.0 == dist);
		dist = tree1.minRDistDual(tree1, 2, tree2, 0);
		assertTrue(7.930000000000001 == dist);
		
		
		tree1 = new BallTree(X1);
		tree2 = new BallTree(X2);
		
		dist = tree1.minRDistDual(tree1, 0, tree2, 0);
		assertTrue(0.0 == dist);
		
		dist = tree1.minRDistDual(tree1, 2, tree2, 0);
		// TODO: assertion
	}
	
	@Test
	public void testMinRDist() {
		Array2DRowRealMatrix X1 = IRIS;
		NearestNeighborHeapSearch tree1 = new KDTree(X1);
		double[] a = new double[]{5.1, 3.5, 1.4, 0.2};
		
		assertTrue(tree1.minRDist(tree1, 1, a) == 0);
		assertTrue(tree1.minRDist(tree1, 2, a) == 10.000000000000004);
		
		a = new double[]{4.9, 3.0, 1.4, 0.2};
		assertTrue(tree1.minRDist(tree1, 1, a) == 0);
		assertTrue(tree1.minRDist(tree1, 2, a) == 10.000000000000004);
	}
	
	@Test
	public void moreNodeHeapTests() {
		NodeHeap nh = new NodeHeap(10);
		assertTrue(nh.n == 0);
		
		nh.push(new NodeHeapData());
		assertTrue(nh.n == 1);
		
		nh.resize(15);
		assertTrue(nh.n == 1);
		
		nh.resize(2);
		assertTrue(nh.n == 1);
		
		
		// Now test some pushes...
		Random seed = new Random(5);
		NodeHeapData node;
		for(int i = 0; i < 10; i++) {
			
			node = new NodeHeapData(
				10.0 - i,
				//seed.nextDouble() * seed.nextInt(40),
				seed.nextInt(5),
				seed.nextInt(100)
			);
			
			nh.push(node);
		}
		
		assertTrue(nh.n == 11);
		nh.pop();
		assertTrue(nh.n == 10);
		assertTrue(nh.peek().val == 1.0);
		
		nh.toString(); // Ensure does not throw NPE
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void nodeHeapResizeUnder1() {
		NodeHeap nh = new NodeHeap(10);
		nh.resize(0); // Here is the exception
	}
	
	
	@Test
	public void testQueryBig() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		double[][] expectedDists = new double[][]{
			new double[]{  0.        ,  0.1       ,  0.14142136},
			new double[]{  0.        ,  0.14142136,  0.14142136},
			new double[]{  0.        ,  0.14142136,  0.24494897},
			new double[]{  0.        ,  0.14142136,  0.17320508},
			new double[]{  0.        ,  0.14142136,  0.17320508},
			new double[]{  0.        ,  0.33166248,  0.34641016},
			new double[]{  0.        ,  0.2236068 ,  0.26457513},
			new double[]{  0.        ,  0.1       ,  0.14142136},
			new double[]{  0.        ,  0.14142136,  0.3       },
			new double[]{  0.        ,  0.        ,  0.        }
		};
		
		int[][] expectedIndices = new int[][]{
			new int[]{ 0, 17,  4},
			new int[]{ 1, 45, 12},
			new int[]{ 2, 47,  3},
			new int[]{ 3, 47, 29},
			new int[]{ 4,  0, 17},
			new int[]{ 5, 18, 10},
			new int[]{ 6, 47,  2},
			new int[]{ 7, 39, 49},
			new int[]{ 8, 38,  3},
			new int[]{37,  9, 34}
		};
		

		// Assert node data equal
		NodeData[] expectedNodeData = new NodeData[]{
			new NodeData(0, 150, false, 10.29635857961444),
			new NodeData(0, 75,  true,  3.5263295365010903),
			new NodeData(75,150, true,  4.506106967216822)
		};
		
		NodeData comparison;
		for(int i = 0; i < expectedNodeData.length; i++) {
			comparison = tree.node_data[i];
			comparison.toString(); // Just to make sure toString() doesn't create NPE
			assertTrue(comparison.equals(expectedNodeData[i]));
		}
		
		
		Neighborhood neighb;
		boolean[] trueFalse = new boolean[]{false, true};
		for(boolean dualTree: trueFalse) {
			
			neighb= tree.query(query, 3, dualTree, true);
			
			assertTrue(MatUtils.equalsWithTolerance(expectedDists, neighb.getDistances(), 1e-8));
			assertTrue(MatUtils.equalsExactly(expectedIndices, neighb.getIndices()));
		}
	}
	
	
	@Test
	public void testQueryRadiusNoSort() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		double[][] expectedNonSortedDists = new double[][]{
			new double[]{ 0.        ,  0.53851648,  0.50990195,  0.64807407,  0.14142136,
			        0.6164414 ,  0.51961524,  0.17320508,  0.46904158,  0.37416574,
			        0.37416574,  0.59160798,  0.54772256,  0.1       ,  0.74161985,
			        0.33166248,  0.43588989,  0.3       ,  0.64807407,  0.46904158,
			        0.59160798,  0.54772256,  0.31622777,  0.14142136,  0.14142136,
			        0.53851648,  0.53851648,  0.38729833,  0.6244998 ,  0.46904158,
			        0.37416574,  0.41231056,  0.46904158,  0.14142136,  0.17320508,
			        0.76811457,  0.45825757,  0.6164414 ,  0.59160798,  0.36055513,
			        0.58309519,  0.3       ,  0.2236068 },
			
			       new double[]{ 0.53851648,  0.        ,  0.3       ,  0.33166248,  0.60827625,
			        0.50990195,  0.42426407,  0.50990195,  0.17320508,  0.45825757,
			        0.14142136,  0.678233  ,  0.54772256,  0.70710678,  0.76157731,
			        0.78102497,  0.55677644,  0.64807407,  0.2236068 ,  0.5       ,
			        0.59160798,  0.5       ,  0.34641016,  0.24494897,  0.678233  ,
			        0.17320508,  0.3       ,  0.78740079,  0.17320508,  0.50990195,
			        0.45825757,  0.52915026,  0.54772256,  0.678233  ,  0.14142136,
			        0.36055513,  0.31622777},
			       
			       new double[]{ 0.50990195,  0.3       ,  0.        ,  0.24494897,  0.50990195,
			        0.26457513,  0.41231056,  0.43588989,  0.31622777,  0.37416574,
			        0.26457513,  0.5       ,  0.51961524,  0.75498344,  0.7       ,
			        0.50990195,  0.64807407,  0.64031242,  0.46904158,  0.50990195,
			        0.6164414 ,  0.54772256,  0.3       ,  0.33166248,  0.78102497,
			        0.31622777,  0.31622777,  0.31622777,  0.36055513,  0.48989795,
			        0.43588989,  0.3       ,  0.65574385,  0.26457513,  0.78102497,
			        0.14142136,  0.33166248},
			       
			       new double[]{ 0.64807407,  0.33166248,  0.24494897,  0.        ,  0.64807407,
			        0.33166248,  0.5       ,  0.3       ,  0.31622777,  0.37416574,
			        0.26457513,  0.51961524,  0.65574385,  0.70710678,  0.64807407,
			        0.53851648,  0.42426407,  0.54772256,  0.72111026,  0.678233  ,
			        0.17320508,  0.2236068 ,  0.31622777,  0.50990195,  0.31622777,
			        0.3       ,  0.58309519,  0.60827625,  0.3       ,  0.7       ,
			        0.26457513,  0.14142136,  0.45825757},
			       
			       new double[]{ 0.14142136,  0.60827625,  0.50990195,  0.64807407,  0.        ,
			        0.6164414 ,  0.45825757,  0.2236068 ,  0.52915026,  0.42426407,
			        0.34641016,  0.64031242,  0.54772256,  0.17320508,  0.79372539,
			        0.26457513,  0.53851648,  0.26457513,  0.56568542,  0.52915026,
			        0.57445626,  0.63245553,  0.34641016,  0.24494897,  0.28284271,
			        0.53851648,  0.57445626,  0.5       ,  0.55677644,  0.78102497,
			        0.52915026,  0.4472136 ,  0.51961524,  0.52915026,  0.24494897,
			        0.17320508,  0.72801099,  0.45825757,  0.58309519,  0.64031242,
			        0.3       ,  0.56568542,  0.33166248,  0.3       },
			       
			       new double[]{ 0.6164414 ,  0.6164414 ,  0.        ,  0.7       ,  0.34641016,
			        0.678233  ,  0.6164414 ,  0.4       ,  0.59160798,  0.33166248,
			        0.38729833,  0.53851648,  0.41231056,  0.678233  ,  0.64807407,
			        0.52915026,  0.64807407,  0.53851648,  0.45825757,  0.47958315,
			        0.60827625,  0.64807407,  0.7       ,  0.60827625,  0.37416574,
			        0.38729833,  0.36055513},
			       
			       new double[]{ 0.51961524,  0.50990195,  0.26457513,  0.33166248,  0.45825757,
			        0.        ,  0.42426407,  0.54772256,  0.47958315,  0.3       ,
			        0.48989795,  0.6164414 ,  0.50990195,  0.64807407,  0.6       ,
			        0.45825757,  0.6244998 ,  0.54772256,  0.60827625,  0.45825757,
			        0.6244998 ,  0.60827625,  0.31622777,  0.42426407,  0.47958315,
			        0.5       ,  0.47958315,  0.46904158,  0.51961524,  0.42426407,
			        0.31622777,  0.54772256,  0.4472136 ,  0.678233  ,  0.2236068 ,
			        0.77459667,  0.42426407},
			       
			       new double[]{ 0.17320508,  0.42426407,  0.41231056,  0.5       ,  0.2236068 ,
			        0.7       ,  0.42426407,  0.        ,  0.78740079,  0.33166248,
			        0.5       ,  0.2236068 ,  0.46904158,  0.7       ,  0.2       ,
			        0.42426407,  0.4472136 ,  0.37416574,  0.67082039,  0.38729833,
			        0.4472136 ,  0.41231056,  0.2236068 ,  0.2236068 ,  0.2236068 ,
			        0.37416574,  0.37416574,  0.4472136 ,  0.73484692,  0.33166248,
			        0.36055513,  0.54772256,  0.33166248,  0.74833148,  0.1       ,
			        0.24494897,  0.66332496,  0.42426407,  0.60827625,  0.46904158,
			        0.42426407,  0.45825757,  0.42426407,  0.14142136},
			       
			       new double[]{ 0.50990195,  0.43588989,  0.3       ,  0.54772256,  0.78740079,
			        0.        ,  0.55677644,  0.67082039,  0.42426407,  0.34641016,
			        0.64031242,  0.46904158,  0.48989795,  0.55677644,  0.7       ,
			        0.55677644,  0.14142136,  0.6244998 ,  0.31622777,  0.42426407,
			        0.36055513,  0.72111026},
			       
			       new double[]{ 0.46904158,  0.17320508,  0.31622777,  0.31622777,  0.52915026,
			        0.47958315,  0.33166248,  0.55677644,  0.        ,  0.78740079,
			        0.34641016,  0.17320508,  0.72801099,  0.5       ,  0.75498344,
			        0.6244998 ,  0.7       ,  0.77459667,  0.52915026,  0.51961524,
			        0.2       ,  0.4472136 ,  0.50990195,  0.4472136 ,  0.26457513,
			        0.17320508,  0.65574385,  0.        ,  0.34641016,  0.75498344,
			        0.        ,  0.55677644,  0.37416574,  0.5       ,  0.55677644,
			        0.65574385,  0.26457513,  0.74161985,  0.34641016,  0.72801099,
			        0.26457513}
		};
		
		int[][] expectedNonSortedIndices = new int[][]{
			new int[]{ 0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 16, 17, 18, 19, 20,
			        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 39,
			        40, 42, 43, 44, 45, 46, 47, 48, 49},
			
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 20, 21, 22, 23,
	        24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40, 42, 43,
	        45, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 19, 21, 22, 23,
	        24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 37, 38, 39, 40, 42, 43, 45,
	        46, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 22, 23, 24, 25,
	        26, 27, 28, 29, 30, 34, 35, 37, 38, 39, 40, 42, 43, 45, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 16, 17, 18, 19, 20,
	        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
	        39, 40, 42, 43, 44, 45, 46, 47, 48, 49},
	        
	        new int[]{ 0,  4,  5,  7, 10, 14, 15, 16, 17, 18, 19, 20, 21, 23, 26, 27, 28,
	        31, 32, 33, 36, 39, 40, 43, 44, 46, 48},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 11, 12, 13, 17, 19, 21, 22, 23,
	        24, 25, 26, 27, 28, 29, 30, 34, 35, 37, 38, 39, 40, 42, 43, 45, 46,
	        47, 48, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 16, 17, 19, 20,
	        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38,
	        39, 40, 42, 43, 44, 45, 46, 47, 48, 49},
	        
	        new int[]{ 1,  2,  3,  6,  7,  8,  9, 11, 12, 13, 25, 29, 30, 34, 35, 37, 38,
	        41, 42, 45, 47, 49},
	        
	        new int[]{ 0,  1,  2,  3,  4,  6,  7,  8,  9, 10, 11, 12, 13, 17, 19, 20, 21,
	        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39, 40,
	        42, 43, 45, 46, 47, 48, 49}
		};
		
		
		
		Neighborhood neighb = tree.queryRadius(query, 0.8, false);
		
		// Just want to know that the total diff in matrices generated from sklearn and clust4j
		// is less than some arbitrarily low number, say one (rounding error).
		assertTrue(absDiffInMatrices(expectedNonSortedDists, neighb.getDistances()) < 1);
		
		int[][] indices = neighb.getIndices();
		for(int i = 0; i < expectedNonSortedIndices.length; i++)
			assertTrue(differenceInIdxArrays(expectedNonSortedIndices[i], indices[i]) <= 2);
	}
	
	@Test
	public void testQueryRadiusWithSort() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		double[][] expectedSortedDists = new double[][]{
			new double[]{ 0.        ,  0.1       ,  0.14142136,  0.14142136,  0.14142136,
	        0.14142136,  0.17320508,  0.17320508,  0.2236068 ,  0.3       ,
	        0.3       ,  0.31622777,  0.33166248,  0.36055513,  0.37416574,
	        0.37416574,  0.37416574,  0.38729833,  0.41231056,  0.43588989,
	        0.45825757,  0.46904158,  0.46904158,  0.46904158,  0.46904158,
	        0.50990195,  0.51961524,  0.53851648,  0.53851648,  0.53851648,
	        0.54772256,  0.54772256,  0.58309519,  0.59160798,  0.59160798,
	        0.59160798,  0.6164414 ,  0.6164414 ,  0.6244998 ,  0.64807407,
	        0.64807407,  0.74161985,  0.76811457},
	       new double[]{ 0.        ,  0.14142136,  0.14142136,  0.17320508,  0.17320508,
	        0.17320508,  0.2236068 ,  0.24494897,  0.3       ,  0.3       ,
	        0.31622777,  0.33166248,  0.34641016,  0.36055513,  0.42426407,
	        0.45825757,  0.45825757,  0.5       ,  0.5       ,  0.50990195,
	        0.50990195,  0.50990195,  0.52915026,  0.53851648,  0.54772256,
	        0.54772256,  0.55677644,  0.59160798,  0.60827625,  0.64807407,
	        0.678233  ,  0.678233  ,  0.678233  ,  0.70710678,  0.76157731,
	        0.78102497,  0.78740079},
	       new double[]{ 0.        ,  0.14142136,  0.24494897,  0.26457513,  0.26457513,
	        0.26457513,  0.3       ,  0.3       ,  0.3       ,  0.31622777,
	        0.31622777,  0.31622777,  0.31622777,  0.33166248,  0.33166248,
	        0.36055513,  0.37416574,  0.41231056,  0.43588989,  0.43588989,
	        0.46904158,  0.48989795,  0.5       ,  0.50990195,  0.50990195,
	        0.50990195,  0.50990195,  0.51961524,  0.54772256,  0.6164414 ,
	        0.64031242,  0.64807407,  0.65574385,  0.7       ,  0.75498344,
	        0.78102497,  0.78102497},
	       new double[]{ 0.        ,  0.14142136,  0.17320508,  0.2236068 ,  0.24494897,
	        0.26457513,  0.26457513,  0.3       ,  0.3       ,  0.3       ,
	        0.31622777,  0.31622777,  0.31622777,  0.33166248,  0.33166248,
	        0.37416574,  0.42426407,  0.45825757,  0.5       ,  0.50990195,
	        0.51961524,  0.53851648,  0.54772256,  0.58309519,  0.60827625,
	        0.64807407,  0.64807407,  0.64807407,  0.65574385,  0.678233  ,
	        0.7       ,  0.70710678,  0.72111026},
	       new double[]{ 0.        ,  0.14142136,  0.17320508,  0.17320508,  0.2236068 ,
	        0.24494897,  0.24494897,  0.26457513,  0.26457513,  0.28284271,
	        0.3       ,  0.3       ,  0.33166248,  0.34641016,  0.34641016,
	        0.42426407,  0.4472136 ,  0.45825757,  0.45825757,  0.5       ,
	        0.50990195,  0.51961524,  0.52915026,  0.52915026,  0.52915026,
	        0.52915026,  0.53851648,  0.53851648,  0.54772256,  0.55677644,
	        0.56568542,  0.56568542,  0.57445626,  0.57445626,  0.58309519,
	        0.60827625,  0.6164414 ,  0.63245553,  0.64031242,  0.64031242,
	        0.64807407,  0.72801099,  0.78102497,  0.79372539},
	       new double[]{ 0.        ,  0.33166248,  0.34641016,  0.36055513,  0.37416574,
	        0.38729833,  0.38729833,  0.4       ,  0.41231056,  0.45825757,
	        0.47958315,  0.52915026,  0.53851648,  0.53851648,  0.59160798,
	        0.60827625,  0.60827625,  0.6164414 ,  0.6164414 ,  0.6164414 ,
	        0.64807407,  0.64807407,  0.64807407,  0.678233  ,  0.678233  ,
	        0.7       ,  0.7       },
	       new double[]{ 0.        ,  0.2236068 ,  0.26457513,  0.3       ,  0.31622777,
	        0.31622777,  0.33166248,  0.42426407,  0.42426407,  0.42426407,
	        0.42426407,  0.4472136 ,  0.45825757,  0.45825757,  0.45825757,
	        0.46904158,  0.47958315,  0.47958315,  0.47958315,  0.48989795,
	        0.5       ,  0.50990195,  0.50990195,  0.51961524,  0.51961524,
	        0.54772256,  0.54772256,  0.54772256,  0.6       ,  0.60827625,
	        0.60827625,  0.6164414 ,  0.6244998 ,  0.6244998 ,  0.64807407,
	        0.678233  ,  0.77459667},
	       new double[]{ 0.        ,  0.1       ,  0.14142136,  0.17320508,  0.2       ,
	        0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 ,  0.2236068 ,
	        0.24494897,  0.33166248,  0.33166248,  0.33166248,  0.36055513,
	        0.37416574,  0.37416574,  0.37416574,  0.38729833,  0.41231056,
	        0.41231056,  0.42426407,  0.42426407,  0.42426407,  0.42426407,
	        0.42426407,  0.42426407,  0.4472136 ,  0.4472136 ,  0.4472136 ,
	        0.45825757,  0.46904158,  0.46904158,  0.5       ,  0.5       ,
	        0.54772256,  0.60827625,  0.66332496,  0.67082039,  0.7       ,
	        0.7       ,  0.73484692,  0.74833148,  0.78740079},
	       new double[]{ 0.        ,  0.14142136,  0.3       ,  0.31622777,  0.34641016,
	        0.36055513,  0.42426407,  0.42426407,  0.43588989,  0.46904158,
	        0.48989795,  0.50990195,  0.54772256,  0.55677644,  0.55677644,
	        0.55677644,  0.6244998 ,  0.64031242,  0.67082039,  0.7       ,
	        0.72111026,  0.78740079},
	       new double[]{ 0.        ,  0.        ,  0.        ,  0.17320508,  0.17320508,
	        0.17320508,  0.2       ,  0.26457513,  0.26457513,  0.26457513,
	        0.31622777,  0.31622777,  0.33166248,  0.34641016,  0.34641016,
	        0.34641016,  0.37416574,  0.4472136 ,  0.4472136 ,  0.46904158,
	        0.47958315,  0.5       ,  0.5       ,  0.50990195,  0.51961524,
	        0.52915026,  0.52915026,  0.55677644,  0.55677644,  0.55677644,
	        0.6244998 ,  0.65574385,  0.65574385,  0.7       ,  0.72801099,
	        0.72801099,  0.74161985,  0.75498344,  0.75498344,  0.77459667,
	        0.78740079}
		};
		
		int[][] expectedSortedIndices = new int[][]{
			new int[]{ 0, 17,  4, 39, 27, 28, 40,  7, 49, 21, 48, 26, 19, 46, 35, 11, 10,
	        31, 36, 20, 43,  9, 34, 37, 23,  2,  6, 29,  1, 30, 25, 16, 47, 24,
	        12, 45, 44,  5, 32,  3, 22, 18, 42},
	        new int[]{ 1, 45, 12, 37, 34,  9, 25, 30, 35,  2, 49,  3, 29, 47,  7, 39, 11,
	        28, 26, 38,  8,  6, 40,  0, 17, 42, 23, 27,  4, 24, 31, 43, 13, 20,
	        21, 22, 36},
	        new int[]{ 2, 47,  3, 45, 12,  6, 42, 29,  1, 35, 37, 34,  9, 49, 30, 38, 11,
	         7, 40,  8, 25, 39, 13,  0, 26,  4, 22, 17, 28, 27, 24, 23, 43, 21,
	        19, 46, 31},
	        new int[]{ 3, 47, 29, 30,  2, 12, 45, 42, 38,  8,  9, 34, 37,  6,  1, 11, 25,
	        49,  7, 35, 13, 24, 26, 39, 40,  0, 23,  4, 17, 28, 43, 22, 27},
	        new int[]{ 4,  0, 17, 40,  7, 39, 27, 19, 21, 28, 46, 49, 48, 26, 11, 10, 35,
	        43,  6, 31,  2, 36, 34, 37,  9, 23, 29, 20, 16, 32, 22, 47, 24, 30,
	        44,  1,  5, 25, 45, 12,  3, 42, 33, 18},
	        new int[]{ 5, 18, 10, 48, 44, 46, 19, 16, 21, 32, 33, 27, 31, 20, 17, 36, 43,
	         0, 15,  4, 28, 26, 39, 14, 23, 40,  7},
	        new int[]{ 6, 47,  2, 11, 42, 29,  3, 30, 49,  7, 40, 45, 22,  4, 26, 38, 37,
	         9, 34, 12, 35, 17,  1,  0, 39,  8, 24, 43, 21, 25, 28, 13, 23, 27,
	        19, 46, 48},
	        new int[]{ 7, 39, 49,  0, 17, 26, 28, 27, 11,  4, 40,  9, 34, 37, 35, 29, 30,
	        21, 23,  2, 25, 19,  1, 46, 48, 43,  6, 24, 20, 31, 47, 45, 12,  3,
	        10, 36, 44, 42, 22,  5, 16, 32, 38,  8},
	        new int[]{ 8, 38,  3, 42, 13, 47, 12, 45,  2, 29, 30,  1,  6,  9, 37, 34, 41,
	        25, 11, 35, 49,  7},
	        new int[]{34, 37,  9,  1, 30, 12, 25, 49, 29, 45,  2,  3,  7, 35, 11, 47, 39,
	        28, 26,  0,  6, 17, 40, 27, 24, 23,  4, 42, 38,  8, 20, 43, 31, 21,
	        48, 13, 46, 19, 36, 22, 10}
		};
		
		
		
		Neighborhood neighb = tree.queryRadius(query, 0.8, true);
		
		// ensure doesn't throw NPE
		assertTrue(null != neighb.toString());
		
		// ensure doesn't throw NPE
		assertTrue(null != neighb.copy());
		
		// Just want to know that the total diff in matrices generated from sklearn and clust4j
		// is less than some arbitrarily low number, say one (rounding error).
		assertTrue(absDiffInMatrices(expectedSortedDists, neighb.getDistances()) < 1);
		
		int[][] indices = neighb.getIndices();
		for(int i = 0; i < expectedSortedIndices.length; i++)
			assertTrue(differenceInIdxArrays(expectedSortedIndices[i], indices[i]) <= 2);
	}
	
	private static double absDiffInMatrices(double[][] expected, double[][] actual) {
		double sumA = 0;
		double sumB = 0;
		for(int i = 0; i < expected.length; i++) {
			sumA += VecUtils.sum(VecUtils.abs(expected[i]));
			sumB += VecUtils.sum(VecUtils.abs(actual[i]));
		}
		
		return FastMath.abs(sumA - sumB);
	}
	
	private static int differenceInIdxArrays(int[] expected, int[] actual) {
		// Check to see if the diff is <= 2
		ArrayList<Integer> aa = new ArrayList<Integer>();
		ArrayList<Integer> bb = new ArrayList<Integer>();
		
		for(int in: expected)
			aa.add(in);
		for(int in: actual)
			bb.add(in);
		
		ArrayList<Integer> larger = aa.size() > bb.size() ? aa : bb;
		ArrayList<Integer> smaller= aa.equals(larger) ? bb : aa;
		larger.removeAll(smaller);
		
		return larger.size();
	}
	
	private void addOne(MutableDouble d) {
		d.value++;
	}
	
	@Test
	public void testMutDouble2() {
		MutableDouble d= new MutableDouble();
		addOne(d);
		assertTrue(d.value == 1);
	}
	
	@Test
	public void testTwoPointCorrelation() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		double[][] query = new double[10][];
		
		int idx = 0;
		for(double[] row: IRIS.getData()) {
			if(idx == query.length)
				break;
			query[idx++] = row; // copied implicitly
		}
		
		int[] corSingle, corDual;
		corSingle = tree.twoPointCorrelation(query, 2.5, false);
		corDual = tree.twoPointCorrelation(query, 2.5, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(542, 10)));
		
		corSingle = tree.twoPointCorrelation(query, 1.5, false);
		corDual = tree.twoPointCorrelation(query, 1.5, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(489, 10)));
		
		corSingle = tree.twoPointCorrelation(query, 25, false);
		corDual = tree.twoPointCorrelation(query, 25, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(1500, 10)));
		
		

		corSingle = tree.twoPointCorrelation(query, 0, false);
		corDual = tree.twoPointCorrelation(query, 0, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(12, 10)));
		
		corSingle = tree.twoPointCorrelation(query, -1, false);
		corDual = tree.twoPointCorrelation(query, -1, true);
		assertTrue(VecUtils.equalsExactly(corSingle, corDual));
		assertTrue(VecUtils.equalsExactly(corSingle, VecUtils.repInt(0, 10)));
		
		// Test a big query now, just to ensure no exceptions are thrown...
		final double[][] X = IRIS.getData();
		tree.twoPointCorrelation(X, -1, false);
		tree.twoPointCorrelation(X, -1, true);
		tree.twoPointCorrelation(X, -1.0);
		tree.twoPointCorrelation(X, new double[]{1,2});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testTwoPointCorrelationExcept1() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.twoPointCorrelation(new double[][]{new double[]{1,2}}, new double[]{1.5});
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testTwoPointCorrelationExcept2() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.twoPointCorrelation(new double[][]{new double[]{1,2}}, 1.5);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void radiusQueryTestDimException() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{1,2}}, 150.0, true);
	}
	
	@Test
	public void radiusQueryTestAllInRadius() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{2.5,2.5,2.5,2.5}}, 150.0, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void radiusQueryTestMPrimeDimMismatch1() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{2.5,2.5,2.5,2.5}}, 
			new double[]{1,2,3,4,5}, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void radiusQueryTestNDimMismatch2() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.queryRadius(new double[][]{new double[]{2.5,2.5,2.5}}, 
			new double[]{5}, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void queryNDimMismatch1() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.query(new double[][]{new double[]{1,2}}, 2, true, true);
	}
	
	@Test(expected=DimensionMismatchException.class)
	public void testKernelDimMismatch() {
		NearestNeighborHeapSearch tree = new KDTree(IRIS);
		tree.kernelDensity(new double[][]{new double[]{1.0}}, 1.0, PartialKernelDensity.LOG_COSINE);
	}
	
	@Test
	public void testNodeDataEquals() {
		NodeData n1 = new NodeData();
		NodeData n2 = new NodeData(1,2,true,1.9);
		
		assertTrue(n1.equals(n1));
		assertFalse(n1.equals(n2));
		assertFalse(n1.equals(new String()));
	}
	
	@Test
	public void testInfDist() {
		Array2DRowRealMatrix mat = new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3), false);
		
		KDTree k = new KDTree(mat, Distance.CHEBYSHEV);
		Neighborhood n = k.query(mat);
		Neighborhood p = k.query(mat, 1, false, true);
		assertTrue(n.equals(p));
		assertTrue(n.equals(n));
		assertFalse(n.equals("asdf"));
		
		Neighborhood res = new Neighborhood(
			new double[][]{
				new double[]{0.0},
				new double[]{0.0},
				new double[]{0.0}
			},	
			
			new int[][]{
				new int[]{0},
				new int[]{1},
				new int[]{2}
			}
		);
		
		assertTrue(n.equals(res));
		final int[] corr = k.twoPointCorrelation(mat.getDataRef(), new double[]{1,2,3});
		assertTrue(VecUtils.equalsExactly(corr, new int[]{3,3,7}));
		assertTrue(k.infinity_dist);
	}
	
	@Test
	public void testWarn() {
		Array2DRowRealMatrix mat = new Array2DRowRealMatrix(
			MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3), false);
		
		KDTree k = new KDTree(mat, new HaversineDistance(), new KMeans(mat,1));
		assertTrue(k.logger.hasWarnings());
	}
	
	@Test
	public void testImmutability() {
		double[][] a = MatUtils.reshape(new double[]{
				1,2,3,4,5,6,7,8,9
			}, 3, 3);
		
		double[][] b = MatUtils.copy(a);
		Array2DRowRealMatrix mat = new Array2DRowRealMatrix(a, false);
		
		KDTree k = new KDTree(mat, Distance.EUCLIDEAN);
		k.query(a);
		
		assertTrue(MatUtils.equalsExactly(b, a)); // assert immutability
	}
}
