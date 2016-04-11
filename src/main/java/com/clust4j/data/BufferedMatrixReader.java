/*******************************************************************************
 *    Copyright 2015, 2016 Taylor G Smith
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 *******************************************************************************/
package com.clust4j.data;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.RecursiveTask;
import java.util.concurrent.RejectedExecutionException;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.util.FastMath;

import com.clust4j.Clust4j;
import com.clust4j.GlobalState;
import com.clust4j.algo.ParallelChunkingTask;
import com.clust4j.except.MatrixParseException;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.log.Log;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Loggable;
import com.clust4j.utils.ArrayFormatter;
import com.clust4j.utils.DeepCloneable;
import com.clust4j.utils.MatUtils;
import com.clust4j.utils.VecUtils;


/**
 * A class for reading a {@link DataSet} from files. If the separator
 * is not provided, the class will estimate the separator. This is based
 * on H2O's CsvParser, but is lighter weight, as clust4j mandates 100%
 * numeric matrices. 
 * 
 * <p>
 * The following byte delimiters are supported for auto-estimation:
 * <ul>
 * <li><tt>0x1</tt> - the default Hive delimiter
 * <li><tt>','</tt>
 * <li><tt>'\t'</tt>
 * <li><tt>';'</tt>
 * <li><tt>'|'</tt>
 * <li><tt>' '</tt>
 * </ul>
 * 
 * <p>
 * The parser will also strip out comments in the head of the file, if found. 
 * Additionally, the following tokens will be coerced to {@link Double#NaN}:
 * 
 * <ul>
 * <li><tt>""</tt>
 * <li><tt>" "</tt>
 * <li><tt>nan</tt> (case insensitive)
 * <li><tt>na</tt> (case insensitive)
 * <li><tt>?</tt>
 * </ul>
 * 
 * <p>
 * The following tokens will be coerced to {@link Double#POSITIVE_INFINITY}:
 * 
 * <ul>
 * <li><tt>inf</tt> (case insensitive)
 * <li><tt>infinity</tt> (case insensitive)
 * </ul>
 * 
 * <p>
 * The following tokens will be coerced to {@link Double#NEGATIVE_INFINITY}:
 * 
 * <ul>
 * <li><tt>-inf</tt> (case insensitive)
 * <li><tt>-infinity</tt> (case insensitive)
 * </ul>
 * 
 * @see <a href="https://github.com/h2oai/h2o-3/blob/master/h2o-core/src/main/java/water/parser/CsvParser.java">h2o</a>
 * @author Taylor G Smith
 */
public class BufferedMatrixReader implements Loggable {
	private boolean hasWarnings = false;
	
	/* Chars to watch for... */
	private static final byte HIVE = 0x1;
	private static final byte COMMA = ',';
	private static final byte TAB = '\t';
	private static final byte CARRIAGE = 13;
	private static final byte LINE_FEED = 10;
	private static final byte SPACE = ' ';
	private static final byte DQUOTE = '"';
	private static final byte SQUOTE = '\'';
	//private static final byte DECIMAL = '.';
	private static final byte GUESS_SEP = -1;
	
	/* More statics */
	static final long LARGEST_DIGIT_NUM = Long.MAX_VALUE/10;
	
	/* Separators to watch for... */
	static final byte[] known_separators = new byte[]{
		HIVE /* Hive - '^A' */, 
		COMMA, 
		';', 
		'|' /* MySql, Sqlite */, 
		TAB, 
		SPACE /* Or multiple spaces... */,
	};
	
	/* Separators that need escaping */
	static final byte[] escapable_separators = new byte[]{
		'|'
	};
	
	/* Comment chars to watch for */
	static final byte[] known_comments = new byte[]{
		'#',
		'%',
		'@'
	};
	
	/* Change to NaN */
	static final String[] nan_strings = new String[]{
		"",
		"nan",
		"na",
		"?", // do we want this?
	};
	
	/* Change to Inf */
	static final String[] pos_inf_strings = new String[]{
		"inf",
		"infinity",
	};
	
	/* Change to neg inf */
	static final String[] neg_inf_strings = new String[]{
		"-inf",
		"-infinity",
	};
	
	/** Helper functions */
	static boolean isEscapable(byte b) { for(byte c: escapable_separators) {if(c==b) return true;} return false; }
	static boolean isEOL(byte chr) { return chr == LINE_FEED || chr == CARRIAGE; }
	static boolean isComment(final byte chr) { 
		for(final byte b: known_comments)
			if(chr == b) 
				return true; 
		return false; 
	}
	
	static boolean isNaN(final String lower) {
		for(String nan: nan_strings)
			if(nan.equals(lower))
				return true;
		
		return false;
	}
	
	static boolean isNegInf(final String lower) {
		for(String inf: neg_inf_strings)
			if(inf.equals(lower))
				return true;
		
		return false;
	}
	
	static boolean isPosInf(final String lower) {
		for(String inf: pos_inf_strings)
			if(inf.equals(lower))
				return true;
		
		return false;
	}

	static byte[] fileToBytes(final File file) throws IOException {
		return Files.readAllBytes(file.toPath());
	}
	
	
	
	/* Instance attributes */
	final MatrixReaderSetup setup;
	
	
	
	
	
	
	/**
	 * Create an instance from a file
	 * @param file
	 * @throws MatrixParseException
	 * @throws IOException
	 */
	public BufferedMatrixReader(final File file) throws MatrixParseException, IOException {
		this(fileToBytes(file));
	}
	
	/**
	 * Create an instance from a file
	 * @param file
	 * @param single_quotes
	 * @throws MatrixParseException
	 * @throws IOException
	 */
	public BufferedMatrixReader(final File file, boolean single_quotes) throws MatrixParseException, IOException {
		this(fileToBytes(file), single_quotes);
	}
	
	/**
	 * Create an instance from a file
	 * @param file
	 * @param sep
	 * @throws MatrixParseException
	 * @throws IOException
	 */
	public BufferedMatrixReader(final File file, byte sep) throws MatrixParseException, IOException {
		this(fileToBytes(file), sep);
	}
	
	/**
	 * Create an instance from a file
	 * @param file
	 * @param sep
	 * @param single_quotes
	 * @throws MatrixParseException
	 * @throws IOException
	 */
	public BufferedMatrixReader(final File file, boolean single_quotes, byte sep) throws MatrixParseException, IOException {
		this(fileToBytes(file), single_quotes, sep);
	}
	
	/**
	 * Create an instance from an array of bytes
	 * @param bits
	 * @throws MatrixParseException
	 */
	public BufferedMatrixReader(byte[] bits) throws MatrixParseException {
		this(new MatrixReaderSetup(bits));
	}
	
	/**
	 * Create an instance from an array of bytes
	 * @param bits
	 * @param single_quotes
	 * @throws MatrixParseException
	 */
	public BufferedMatrixReader(byte[] bits, boolean single_quotes) throws MatrixParseException {
		this(new MatrixReaderSetup(bits, single_quotes));
	}
	
	/**
	 * Create an instance from an array of bytes
	 * @param bits
	 * @param single_quotes
	 * @param sep
	 * @throws MatrixParseException
	 */
	public BufferedMatrixReader(byte[] bits, boolean single_quotes, byte sep) throws MatrixParseException {
		this(new MatrixReaderSetup(bits, single_quotes, sep));
	}
	
	/**
	 * Create an instance from an array of bytes and a separator
	 * @param bits
	 * @param sep
	 * @throws MatrixParseException
	 */
	public BufferedMatrixReader(byte[] bits, byte sep) throws MatrixParseException {
		this(new MatrixReaderSetup(bits, sep));
	}
	
	/**
	 * Create an instance from an existing setup
	 * @param setup
	 * @throws MatrixParseException
	 */
	protected BufferedMatrixReader(MatrixReaderSetup setup) throws MatrixParseException {
		this.setup = setup;
		this.hasWarnings = setup.hasWarnings();
	}
	
	
	
	
	
	
	
	/**
	 * A class that guesses the setup of the input file, including
	 * separators, etc.
	 * @author Taylor G Smith
	 */
	protected static class MatrixReaderSetup extends Clust4j implements Loggable, DeepCloneable {
		private static final long serialVersionUID = 5863624610174664028L;
		private static final int GUESS_LINES = 4;
		
		/* Instance vars */
		boolean single_quotes; // whether single quotes quote a field or double quotes do
		final int num_cols;
		int header_offset = 0; // which row to start on due to headers
		String[] headers = null;
		String[][] data; // First few rows of parsed data
		final byte separator;
		final byte[] stream;
		private boolean hasWarnings;
		final LogTimer timer;
		
		/**
		 * Copy constructor
		 * @param instance
		 */
		private MatrixReaderSetup(MatrixReaderSetup instance) {
			this.single_quotes = instance.single_quotes;
			this.num_cols = instance.num_cols;
			this.header_offset = instance.header_offset;
			this.headers = VecUtils.copy(instance.headers); // if null, sets to null
			this.data = MatUtils.copy(instance.data);
			this.separator = instance.separator;
			this.stream = Arrays.copyOf(instance.stream, instance.stream.length);
			this.hasWarnings = instance.hasWarnings;
			this.timer = instance.timer;
		}
		
		MatrixReaderSetup(byte[] bits) throws MatrixParseException {
			this(bits, false, GUESS_SEP);
		}
		
		MatrixReaderSetup(byte[] bits, boolean single_quotes) throws MatrixParseException {
			this(bits, single_quotes, GUESS_SEP);
		}
		
		MatrixReaderSetup(byte[] bits, byte sep) throws MatrixParseException {
			this(bits, false, sep);
		}
		
		MatrixReaderSetup(byte[] bits, boolean single_quotes, byte sep) throws MatrixParseException {
			this.single_quotes = single_quotes;
			if(single_quotes)
				info("using single quotes (\"'\")");
			else info("using double quotes ('\"')");
			
			this.timer = new LogTimer();
			
			/* Given the bytes, we look at first few lines and guess the setup... */
			String[] lines = getFirstLines(bits);
			
			// If data is empty, fail
			if(lines.length == 0)
				error(new MatrixParseException("data is empty!"));
			
			// Guess separator, columns and header
			data = new String[lines.length][];
			
			// Corner case first:
			if( 1 == lines.length ) {
				warn("only one line found in data");
				String line = lines[0];
				
				if(GUESS_SEP == sep) {
					/*
					 *  Guess the separator. Harder to do with only one line
					 */
					
					String splitter;
					boolean foundSep = false;
					
					
					for(byte ks: known_separators) {
						
						/*
						 * Some chars require escaping or they'll
						 * falsely flag their presence.
						 */
						splitter = isEscapable(ks) ?
							new String(new byte[]{(byte)'\\',ks}) :
								new String(new byte[]{ks});
						
						if( line.split(splitter).length > 1 ) {
							foundSep = true;
							sep = ks;
							break;
						}
						
						
						/*
						 *  There's a corner case here... imagine the row is:
						 *  
						 *  "a,b,c"|"d,e,f"
						 *  
						 *  ... since this is ordinally dependent, it will select
						 *  the comma as the separator, though in cases where this
						 *  would happen, we'd likely fail the case on the basis that
						 *  it's text. However, this is a very real possibility:
						 *  
						 *  "10,123"|"12,198"
						 *  
						 *  ... in which case the | should be the delimiter and 
						 *  we selected the wrong one. The moral of the story (two):
						 *  
						 *  - Don't try to read a single-row CSV
						 *  - Don't include thousands separators in your data
						 */
					}
					
					if(!foundSep) { // probably one item
						// If there's one item, we're just going to fail out
						error(new MatrixParseException("could not find separator in row: " + line));
					}
				}
				
				
				/*
				 * One way or another at this point, we have a separator picked out
				 */
				data[0] = getTokens(line, sep, single_quotes);
				this.num_cols = data[0].length;
				
				// What about the header? Always check...
				if(allStrings(data[0]) && !data[0][0].isEmpty()) {
					error(new MatrixParseException("singular "
						+ "row is entirely character; maybe "
						+ "an orphaned header?"));
				}
				
			} else { // 2+ lines
				
				// First guess the separator
				if(GUESS_SEP == sep) {
					sep = guessSeparator(lines[0], lines[1], single_quotes, this);
					
					// extremely difficult-to-replicate corner case... let's keep it simple
					/*
					if(GUESS_SEP == sep && lines.length > 2) {
						sep = guessSeparator(lines[1], lines[2], single_quotes);
						if(GUESS_SEP == sep)
							sep = guessSeparator(lines[0], lines[2], single_quotes);
					}
					
					if(GUESS_SEP == sep) {
						warn("could not determine uniform separator; using space (' ')");
						sep = SPACE; // bail and go for space...
					} else {
						info("separator estimated as '"+new String(new byte[]{sep})+"'");
					}
					*/
					
					if(GUESS_SEP == sep) {
						error(new MatrixParseException("cannot determine uniform separator"));
					} else {
						info("separator estimated as '"+new String(new byte[]{sep})+"'");
					}
				} else {
					info("separator provided as '"+new String(new byte[]{sep})+"'");
				}
				
				
				// Tokenize first few
				for(int i = 0; i < lines.length; ++i)
					data[i] = getTokens(lines[i], sep, single_quotes);
				
				// Guess the number of columns
				this.num_cols = guessNumCols(data);
				
				// Check for header
				if(allStrings(data[0]) && !data[0][0].isEmpty()) {
					header_offset = 1;
					this.headers = data[0];
				}
			}
			
			/*
			 *  Now we need to go through each row and ensure it's 
			 *  completely numeric... this only looks through the first
			 *  few, but gives us confidence, and saves us time later
			 *  if it's bad up front.
			 */
			for(int i = header_offset; i < data.length; i++) {
				try {
					tokenize(data[i]);
				} catch(NumberFormatException e) {
					error(new MatrixParseException("non-numeric row found: " 
						+ ArrayFormatter.arrayToString(data[i])));
				}
			}
			
			// Num cols?
			info(num_cols + " feature"+(num_cols==1?"":"s")+" identified in dataset");
			
			
			this.stream = bits;
			this.separator = sep;
			sayBye(timer);
		}
		
		static boolean allStrings(String[] row) {
			for(String s: row) {
				try {
					Double.parseDouble(s);
					return false;
				} catch(NumberFormatException e) {
				}
			}
			
			return true;
		}
		
		/**
		 * Adapted from H2O's getFirstLines method
		 * @param bits
		 * @return
		 */
		static String[] getFirstLines(byte[] bits) {
			return getLines(bits, GUESS_LINES);
		}
		
		static int[] getSeparatorCounts(String l1, final byte single) {
			// This is essentially a lightweight map... byte : int
			int[] result = new int[known_separators.length];
			byte[] bits  = l1.getBytes();
			
			boolean inQuote = false;
			for(byte c: bits) {
				if(single == c || DQUOTE == c)
					inQuote ^= true; // toggles on or off
				
				if(!inQuote || HIVE == c) {
					for(int i = 0; i < known_separators.length; ++i)
						if(known_separators[i] == c)
							++result[i];
				}
			}
			
			return result;
		}
		
		static int guessNumCols(String[][] data) {
			int longest = 0;
			for(String[] s: data)
				if(s.length > longest)
					longest = s.length;
			
			if(longest == data[0].length)
				return longest; // 1st line longer than or equal to rest, so take it
			
			int lengths[] = new int[longest+1];
			for(String[] s: data)
				lengths[s.length]++;
			
			int maxCnt = 0;
			for(int i = 0; i <= longest; i++)
				if(lengths[i] > lengths[maxCnt])
					maxCnt = i;
			
			return maxCnt;
		}
		
		static byte guessSeparator(String l1, String l2, boolean single_quotes, Loggable logger) {
			final byte single = single_quotes ? SQUOTE : -1;
			int[] s1 = getSeparatorCounts(l1, single);
			int[] s2 = getSeparatorCounts(l2, single);
			
			// If both lines have the same number of separators, it's 
			// likely that one... Separators ordered by likelihood.
			int max = 0;
			
			for(int i = 0; i < s1.length; ++i) {
				if(s1[i] == 0) // didn't show up in this string
					continue;
				if(s1[max] < s1[i]) // new max
					max = i;
				if(s1[i] == s2[i]) { // equal counts
					try {
						logger.trace("trying to separate using '" + (char)known_separators[i] + "'");
						String[] t1 = getTokens(l1, known_separators[i], single);
						String[] t2 = getTokens(l2, known_separators[i], single);
						
						if(t1.length != s1[i]+1 || t2.length != s2[i]+1) // non-uniform
							continue;
						
						return known_separators[i];
					} catch(Exception e) {
						// we ignore this and try another one...
					}
				}
			}
			
			// No separators appeared or we didn't see any equal ones...
			/*// if no uniform separators, just going to bail out with exception
			if(s1[max] == 0) { // try the last one (space) 
				max = known_separators.length - 1;
			} if(s1[max] != 0) {
				String[] t1 = getTokens(l1, known_separators[max], single);
				String[] t2 = getTokens(l2, known_separators[max], single);
				
				if(t1.length == s1[max]+1 && t2.length == s2[max]+1 
					&& t1.length == t2.length) // they are equally split
					return known_separators[max];
			}
			*/
			
			return GUESS_SEP;
		}

		@Override public void error(String msg) {
			Log.err(getLoggerTag(), msg);
		}
		
		@Override public void error(RuntimeException thrown) {
			error(thrown.getMessage());
			throw thrown;
		}

		@Override public void warn(String msg) {
			hasWarnings = true;
			Log.warn(getLoggerTag(), msg);
		}
		
		@Override public void info(String msg) {
			Log.info(getLoggerTag(), msg);
		}
		
		@Override public void trace(String msg) {
			Log.trace(getLoggerTag(), msg);
		}
		
		@Override public void debug(String msg) {
			Log.debug(getLoggerTag(), msg);
		}

		@Override public void sayBye(LogTimer timer) {
			info("matrix parse setup completed in " + timer.toString());
		}

		@Override public Algo getLoggerTag() {
			return parserLoggerTag();
		}

		@Override public boolean hasWarnings() {
			return hasWarnings;
		}

		@Override
		public MatrixReaderSetup copy() {
			return new MatrixReaderSetup(this);
		}
	} // end setup class
	
	
	
	
	static String[] getLines(byte[] bits) {
		return getLines(bits, GlobalState.MAX_ARRAY_SIZE);
	}
	
	static String[] getLines(byte[] bits, int num) {
		ArrayList<String> lines = new ArrayList<>();
		
		int nlines = 0, offset = 0;
		while(offset < bits.length && nlines < num) {
			int lineStart = offset;
			while(offset < bits.length && !isEOL(bits[offset]))
				++offset;
			
			int lineEnd = offset++;
			
			/*
			 *  Windows needs to skip a trailing line-feed 
			 *  char after a carriage return
			 */
			if(offset < bits.length && bits[offset] == LINE_FEED)
				++offset;
			
			// Check for comments at top of dataset
			if(isComment(bits[lineStart]))
				continue;
			
			// Do work
			if(lineEnd > lineStart) {
				String data = new String(bits, lineStart, lineEnd - lineStart).trim();
				if(!data.isEmpty()) {
					lines.add(data);
					nlines++;
				}
			}
		}
		
		return lines.toArray(new String[lines.size()]);
	}
	
	static String[] getTokens(String from, byte sep, boolean single_quotes) {
		final byte single = single_quotes ? SQUOTE : -1;
		return getTokens(from, sep, single);
	}
	
	static String[] getTokens(String from, byte sep, final byte single) {
		final ArrayList<String> tokens = new ArrayList<>();
		byte[] bits = from.getBytes();
		
		int offset = 0;
		int quotes = 0;
		
		while(offset < bits.length) {
			while(offset < bits.length && bits[offset] == SPACE) // skip leading ws
				++offset;
			
			if(offset == bits.length)
				break; // reached end of string
			
			StringBuilder t = new StringBuilder();
			byte c = bits[offset];
			
			if(DQUOTE == c || single == c) {
				quotes = c;
				++offset;
			}
			
			while(offset < bits.length) {
				c = bits[offset];
				
				if(quotes == c) {
					++offset;
					
					if(offset < bits.length && bits[offset] == c) {
						t.append((char)c);
						++offset;
						continue;
					}
					
					quotes = 0;
				} else if(0 == quotes && sep == c || isEOL(c)) {
					break; // break inner only
				} else if(sep != COMMA && c == COMMA) { 
					/*
					 * This is a corner case where the separator is NOT
					 * a comma, but the data may contain thousands separators
					 * and this prevents non-numeric exceptions later.
					 */
					++offset;
					continue;
				} else {
					t.append((char)c);
					++offset;
				}
			}
			
			c = (offset == bits.length) ? LINE_FEED : bits[offset];
			tokens.add(t.toString());
			
			if(isEOL(c) || offset == bits.length)
				break;
			if(c != sep)
				return new String[0]; // error!
			++offset;
		}
		
		// Catch case where last char is a separator, indicating empty last col
		if(bits.length > 0 && bits[bits.length - 1] == sep && bits[bits.length - 1] != SPACE) {
			tokens.add("");
		}
		
		return tokens.toArray(new String[tokens.size()]);
	}
	
	/**
	 * A class for parallel reading in of files
	 * @author Taylor G Smith
	 */
	static class ParallelChunkParser extends RecursiveTask<double[][]> {
		private static final long serialVersionUID = 8556857221656513389L;
		private ArrayList<InstanceChunk> chunks;
		private double[][] result;
		final MatrixReaderSetup setup;
		final int n, hi, lo;
		
		/**
		 * A chunk of instances to parse
		 * @author Taylor G Smith
		 */
		final static class InstanceChunk {
			final String[] rows;
			final int startIdx;
			
			InstanceChunk(String[] rows, int startIdx) {
				this.rows = rows;
				this.startIdx = startIdx;
			}
		}
		
		
		public ParallelChunkParser(ParallelChunkParser instance, int lo, int hi) {
			this.chunks = instance.chunks;
			this.result = instance.result;
			this.setup  = instance.setup;
			this.n = instance.n;
			this.lo = lo;
			this.hi = hi;
		}
		
		private ParallelChunkParser(String[] rows, MatrixReaderSetup setup) {
			this.setup = setup;
			this.n = setup.num_cols;
			this.result = new double[rows.length][n];
			this.chunks = map(rows);
			this.lo = setup.header_offset;
			this.hi = this.chunks.size();
		}

		/**
		 * Given a chunk number, read the chunk
		 * @param chunk
		 * @param startIdx
		 */
		void doChunk(int chunk) {
			final InstanceChunk c = chunks.get(chunk);
			
			int idx = c.startIdx;
			double[] next;
			for(String instance: c.rows) {
				String[] a = getTokens(instance, setup.separator, setup.single_quotes);
				
				try {
					next = tokenize(a);
					
					// Ensure not jagged
					if(next.length != setup.num_cols)
						throw new DimensionMismatchException(next.length, setup.num_cols);
					
					result[idx++] = next;
				} catch(NumberFormatException e) {
					throw new NumberFormatException(ArrayFormatter.arrayToString(a));
				} catch(DimensionMismatchException d) {
					throw d; // propagate it
				} catch(Exception e) {
					throw new RuntimeException("unexpected exception in parallel processing",e);
				}
			}
		}

		@Override
		protected double[][] compute() {
			if(hi - lo <= 1) { // generally should equal one...
				doChunk(lo);
				return result;
			} else {
				int mid = this.lo + (this.hi - this.lo) / 2;
				ParallelChunkParser left = new ParallelChunkParser(this, lo, mid);
				ParallelChunkParser right= new ParallelChunkParser(this, mid,hi );
				
				left.fork();
				right.compute();
				left.join();
				
				return result;
			}
		}
		
		protected static InstanceChunk getChunk(String[] X, int chunkSize, int chunkNum, int header_offset) {
			String[] chunk;
			
			int idx = 0;
			int startingPt = chunkNum * chunkSize + (chunkNum == 0 ? header_offset : 0);
			int endingPt = FastMath.min(X.length, startingPt + chunkSize);
			
			chunk = new String[endingPt - startingPt];
			for(int j = startingPt; j < endingPt; j++) {
				chunk[idx++] = X[j];
			}
			
			return new InstanceChunk(chunk, startingPt);
		}
		
		private ArrayList<InstanceChunk> map(String[] rows) {
			final ArrayList<InstanceChunk> out = new ArrayList<>();
			final int chunkSize = ParallelChunkingTask.ChunkingStrategy.getChunkSize(rows.length);
			final int numChunks = ParallelChunkingTask.ChunkingStrategy.getNumChunks(chunkSize, rows.length);
			
			for(int i = 0; i < numChunks; i++)
				out.add(getChunk(rows, chunkSize, i, this.setup.header_offset));
			
			return out;
		}
		
		public static double[][] doAll(String[] rows, MatrixReaderSetup setup) {
			return GlobalState.ParallelismConf.FJ_THREADPOOL
				.invoke(new ParallelChunkParser(rows, setup));
		}
	}
	
	
	/**
	 * Read in the data
	 * @return the matrix
	 * @throws MatrixParseException
	 */
	public DataSet read() throws MatrixParseException {
		return read(false);
	}
	
	/**
	 * Read in the data
	 * @param parallel - whether to parallelize the operation
	 * @return the matrix
	 * @throws MatrixParseException
	 */
	public DataSet read(boolean parallel) throws MatrixParseException {
		LogTimer timer = new LogTimer();
		String msg;
		
		
		/*
		 * Get lines...
		 */
		String[] lines = getLines(setup.stream);
		
		
		// Potential for truncation here...
		if(lines.length == GlobalState.MAX_ARRAY_SIZE)
			warn("only " + lines.length + " rows read from data, "
				+ "as this is the max clust4j allows");
		else
			info((lines.length-setup.header_offset) + " record"
				+ (lines.length==1?"":"s") + " (" + setup.stream.length
				+ " byte"+(setup.stream.length==1?"":"s")+") read from file");
		
		
		/*
		 * Do double parsing...
		 */
		double[][] res = null;
		if(!parallel) {
			// Let any exceptions propagate
			res = parseSerial(lines);
		} else {
			
			boolean throwing_exception = true;
			try {
				res = ParallelChunkParser.doAll(lines, setup);
			} catch(NumberFormatException n) {
				error(new MatrixParseException("caught NumberFormatException: " + n.getLocalizedMessage()));
			} catch(DimensionMismatchException d) {
				error(new MatrixParseException("caught row of unexpected dimensions: " + d.getMessage()));
			} catch(RejectedExecutionException r) {
				throwing_exception = false;
				warn("unable to schedule parallel job; falling back to serial parse");
				res = parseSerial(lines);
			} catch(Exception e) {
				msg = "encountered Exception in thread" + e.getMessage();
				error(msg);
				throw e;
			} finally {
				if(null == res && !throwing_exception)
					throw new RuntimeException("unable to parse data");
			}
		}
		
		
		sayBye(timer);
		return new DataSet(res, setup.headers);
	}
	
	private double[][] parseSerial(String[] lines) {
		int k = 0;
		String msg, line;
		double[] next;
		
		double[][] res = new double[lines.length - setup.header_offset][setup.num_cols];
		for( int idx = setup.header_offset; idx < lines.length; idx++ ) {
			line = lines[idx];
			
			try {
				next = tokenize(line);
				
				// Ensure not jagged
				if(next.length != setup.num_cols) {
					msg = "expected row of length " + setup.num_cols + 
						"; got row of length " + next.length + " at line " +
						idx;
					error(msg);
					throw new MatrixParseException(msg);
				}
				
				res[k++] = next;
			} catch(NumberFormatException e) {
				msg = "non-numeric row found: " + line;
				error(msg);
				throw new MatrixParseException(msg);
			}
		}
		
		return res;
	}
	
	/**
	 * Handle the tokenizing logic for this instance
	 * @param row
	 * @return
	 * @throws NumberFormatException
	 */
	private double[] tokenize(String row) throws NumberFormatException {
		final String[] tokens = getTokens(row, setup.separator, setup.single_quotes);
		return tokenize(tokens);
	}
	
	/**
	 * Static tokenizing method to move a row of strings into a double array
	 * @param row
	 * @return
	 * @throws NumberFormatException
	 */
	static double[] tokenize(String[] row) throws NumberFormatException {
		final double[] out = new double[row.length];
		
		int idx = 0;
		for(String str: row) {
			double val = 0;
			
			try {
				val = Double.parseDouble(str);
			} catch(NumberFormatException e) {
				String lower = str.toLowerCase();
				
				// Check if it's a nan...
				if(isNaN(lower))
					val = Double.NaN;
				else if(isPosInf(lower))
					val = Double.POSITIVE_INFINITY;
				else if(isNegInf(lower))
					val = Double.NEGATIVE_INFINITY;
				else
					throw e;
			}
			
			out[idx++] = val;
		}
		
		return out;
	}
	

	@Override public void error(String msg) {
		Log.err(getLoggerTag(), msg);
	}
	
	@Override public void error(RuntimeException thrown) {
		error(thrown.getMessage());
		throw thrown;
	}

	@Override public void warn(String msg) {
		hasWarnings = true;
		Log.warn(getLoggerTag(), msg);
	}
	
	@Override public void info(String msg) {
		Log.info(getLoggerTag(), msg);
	}
	
	@Override public void trace(String msg) {
		Log.trace(getLoggerTag(), msg);
	}
	
	@Override public void debug(String msg) {
		Log.debug(getLoggerTag(), msg);
	}

	@Override public void sayBye(LogTimer timer) {
		info("dataset parsed from file in " + timer.toString());
	}

	@Override public Algo getLoggerTag() {
		return parserLoggerTag();
	}

	@Override public boolean hasWarnings() {
		return hasWarnings;
	}
	
	/**
	 * Gets called from Setup class as well
	 * @return
	 */
	final static Algo parserLoggerTag() {
		return Algo.PARSER;
	}
}
