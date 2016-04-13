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
package com.clust4j.log;

import static com.clust4j.log.Log.Tag.*;

import java.io.*;
import java.net.URI;
import java.util.ArrayList;
import java.util.Locale;

import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.apache.log4j.PropertyConfigurator;

/**
 * A wrapper class for log4j adapted heavily from 0XData H2O's logger class
 * @author Taylor G Smith, with many adaptations and class methods/inner classes from 
 * <a href="https://github.com/h2oai/h2o-2/blob/master/src/main/java/water/util/Log.java">H2O Log</a>
 *
 */
public abstract class Log {
	
	/** Tags for log messages */
	public static interface Tag {
		/** Which algorithm is being run? */
		public static enum Algo implements Tag {
			AFFINITY_PROP	{ @Override public String toString(){return "AFFINTY";} },
			AGGLOMERATIVE 	{ @Override public String toString(){return "AGGLOM ";} },
			CLUST4J,
			
			/** To be used with any custom user cluster algo extensions... */
			CUSTOM			{ @Override public String toString(){return "CUSTOM ";} },
			DBSCAN 			{ @Override public String toString(){return "DBSCAN ";} },
			HDBSCAN 		{ @Override public String toString(){return "HDBSCAN";} },
			
			/** Used for matrix imputations */
			IMPUTE			{ @Override public String toString(){return "IMPUTE ";} },
			
			/** More algos... */
			KMEDOIDS		{ @Override public String toString(){return "KMEDOID";} },
			KMEANS 			{ @Override public String toString(){return "K-MEANS";} },
			MEANSHIFT		{ @Override public String toString(){return "MNSHIFT";} },
			NEAREST			{ @Override public String toString(){return "NEAREST";} },
			RADIUS          { @Override public String toString(){return "RADIUS ";} },
			
			/*
			 * For file parsing...
			 */
			PARSER			{ @Override public String toString(){return "PARSER ";} },
			
			;
			
			
			boolean _enable;
		}
		
		/** What kind of message to log */
		public static enum Type implements Tag {
			TRACE, 
			DEBUG, 
			
			// add a space to the four-letter words
			INFO	{ @Override public String toString(){return "INFO ";} }, 
			WARN	{ @Override public String toString(){return "WARN ";} }, 
			ERROR, 
			FATAL
		}
	}
	
	
	final static public Timer theTimer = new LogTimer();
	
	
	
	
	
	
	
	/**
	 * PrintStream wrapper
	 * @author 0xData
	 */
	final static class LogWrapper extends PrintStream {
		PrintStream parent;
		
		LogWrapper(PrintStream parent) {
			super(parent);
			this.parent = parent;
		}
		
		private static String log(Locale l, boolean nl, String format, Object... args) {
			String msg = String.format(l, format, args);
			LogEvent e = LogEvent.make(Algo.CLUST4J, Type.INFO, null, msg);
			Log.write(e, false); // Skip the KVLog present in H2O
			return e.toShortString() + lineSep;
		}
		
		@Override
		public PrintStream printf(String format, Object... args) {
			super.printf(log(null, false, format, args));
			return this;
		}
		
		@Override
		public PrintStream printf(Locale l, String format, Object... args) {
			super.printf(log(l, false, format, args));
			return this;
		}
		
		@Override
		public void println(String x) {
			super.print(log(null, true, "%s", x));
		}
		
		void printlnParent(String s) {
			super.println(s);
		}
	}
	
	
	
	
	
	
	
	/**
	 * 0XData Event class
	 */
	static class LogEvent {
		Type type;
		Algo algo;
		Timer when;
		long msFromStart;
		Throwable ouch;
		Object[] messages;
		Object message;
		String thread;
		
		volatile boolean printMe;
		
		/* These are all volatile in H2O's API */
		private volatile static Timer lastGoodTimer = new LogTimer();
		private volatile static LogEvent lastEvent = new LogEvent();
		private volatile static int missed;
		
		/* Builder methods */
		static LogEvent make(Tag.Algo algo, Tag.Type type, Throwable ouch, Object[] messages) {
			return make0(algo, type, ouch, messages, null);
		}
		
		static LogEvent make(Tag.Algo algo, Tag.Type type, Throwable ouch, Object message) {
			return make0(algo, type, ouch, null, message);
		}
		
		static private LogEvent make0(
				Tag.Algo algo, Tag.Type type, Throwable ouch, 
				Object[] messages, Object message) {
			LogEvent result = null;
			
			try {
				result = new LogEvent();
				result.init(algo, type, ouch, messages, message, lastGoodTimer=new LogTimer());
			} catch(OutOfMemoryError e) {
				synchronized(LogEvent.class) {
					if(lastEvent.printMe) {
						missed++;
						return null;
					}
					
					result = lastEvent;
					result.init(algo, type, ouch, messages, null, lastGoodTimer);
				}
			}
			
			return result;
		}
		
		private void init(Tag.Algo algo, Tag.Type type, 
				Throwable ouch, Object[] messages, 
				Object message, Timer timer) {
			this.algo = algo;
			this.type = type;
			this.ouch = ouch;
			this.messages = messages;
			this.message = message;
			this.when = timer;
			this.printMe = true;
		}
		
		@Override
		public String toString() {
			StringBuilder sb = longHeader(new StringBuilder(120));
			int headroom = sb.length();
			sb.append(body(headroom));
			return sb.toString();
		}
		
		public String toShortString() {
			StringBuilder sb = shortHeader(new StringBuilder(120));
			int headroom = sb.length();
			sb.append(body(headroom));
			return sb.toString();
		}
		
		public String body(final int headroom) {
			StringBuilder buf= new StringBuilder(120);
			
			// If there are messages...
			if(messages != null) {
				for(Object m: messages)
					buf.append(m.toString());
			} else if(message != null)
				buf.append(message.toString());
			
			// --- A NOTE FROM THE H2O DEVELOPERS: ---
			// --- "\n" vs lineSep ---
			// Embedded strings often use "\n" to denote a new-line.  This is either
			// 1 or 2 chars ON OUTPUT depending Unix vs Windows, but always 1 char in
			// the incoming string.  We search & split the incoming string based on
			// the 1 character "\n", but we build result strings with lineSep (a 
			// String of length 1 or 2).  i.e.
			// GOOD: String.indexOf("\n"); SB.append( lineSep )
			// BAD : String.indexOf( lineSep ); SB.append("\n")
			
			if(buf.indexOf("\n") != -1) {
				String[] lines = buf.toString().split("\n");
				
				if(lines.length > 0) {
					StringBuilder buf2 = new StringBuilder(2 * buf.length());
					buf2.append(lines[0]);
					
					for(int i = 1; i < lines.length; i++) {
						buf2.append(lineSep).append("+");
						for(int j = 1; j < headroom; j++)
							buf2.append(" ");
						buf2.append(lines[i]);
					}
					
					buf = buf2;
				}
			}
			
			// Handle any throwables...
			if(null != ouch) {
				buf.append(lineSep);
				Writer wr = new StringWriter();
				PrintWriter pwr = new PrintWriter(wr);
				ouch.printStackTrace(pwr);
				
				String mess = wr.toString();
				String[] lines = mess.split("\n");
				for(int i = 0; i < lines.length; i++) {
					buf.append("+");
					for(int j = 1; j < headroom; j++)
						buf.append(" ");
					buf.append(lines[i]);
					if( i != lines.length - 1 )
						buf.append(lineSep);
				}
			}
			
			return buf.toString();
		}
		
		private StringBuilder longHeader(StringBuilder buf) {
			buf.append(when.startAsString()).append(" ");
			buf.append(type.toString()).append(" ").append(algo.toString()).append(": ");
			return buf;
		}
		
		/**
		 * In the H2O API, the difference is this won't append threadnames. Since
		 * this version is non-concurrent and there are no threads anyways, we will
		 * only return the longHeader(StringBuilder)
		 * @param buf
		 * @return
		 */
		private StringBuilder shortHeader(StringBuilder buf) {
			return longHeader(buf);
		}
	}
	
	
	
	
	
	/* Main write method */
	private static void write(LogEvent e, boolean printOnOut) {
		try {
			write0(e, printOnOut);
			
			if(LogEvent.lastEvent.printMe || LogEvent.missed > 0) {
				
				synchronized(LogEvent.class) {
					if(LogEvent.lastEvent.printMe) {
						LogEvent ev = LogEvent.lastEvent;
						write0(ev, true);
						LogEvent.lastEvent = new LogEvent();
					}
					
					if(LogEvent.missed > 0 && !LogEvent.lastEvent.printMe) {
						LogEvent.lastEvent.init(Algo.CLUST4J, Type.WARN, null, null, "Logging framework dropped a message", LogEvent.lastGoodTimer);
						LogEvent.missed--;
					}
				}
				
			}
			
		} catch(OutOfMemoryError xe) {
			synchronized(LogEvent.class) {
				if(!LogEvent.lastEvent.printMe)
					LogEvent.lastEvent = e;
				else LogEvent.missed++;
			}
		}
	}
	
	
	
	
	/**
	 * The main logger...
	 */
	protected static org.apache.log4j.Logger _logger = null;
	
	
	public static String getLogDir() {
		if(null == LOG_DIR)
			return "unknown-log-dir";
		return LOG_DIR;
	}
	
	public static String getLogPathFileNameStem() {
		return getLogDir() + File.separator + "clust4j";
	}
	
	public static String getLogPathFileName() {
		return getLogPathFileNameStem() + "-debug.log";
	}
	
	private static org.apache.log4j.Logger getLog4jLogger() {
		return _logger;
	}
	
	private static void setLog4jProperties(String logDirParent, java.util.Properties p) {
	    LOG_DIR = logDirParent + File.separator + "clust4jlogs";
	    String logPathFileName = getLogPathFileNameStem();

	    // clust4j-wide logging
	    p.setProperty("log4j.rootLogger", "TRACE, R1, R2, R3, R4, R5, R6");

	    p.setProperty("log4j.appender.R1",                          "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.R1.Threshold",                "TRACE");
	    p.setProperty("log4j.appender.R1.File",                     logPathFileName + "-1-trace.log");
	    p.setProperty("log4j.appender.R1.MaxFileSize",              "1MB");
	    p.setProperty("log4j.appender.R1.MaxBackupIndex",           "3");
	    p.setProperty("log4j.appender.R1.layout",                   "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.R1.layout.ConversionPattern", "%m%n");

	    p.setProperty("log4j.appender.R2",                          "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.R2.Threshold",                "DEBUG");
	    p.setProperty("log4j.appender.R2.File",                     logPathFileName + "-2-debug.log");
	    p.setProperty("log4j.appender.R2.MaxFileSize",              "3MB");
	    p.setProperty("log4j.appender.R2.MaxBackupIndex",           "3");
	    p.setProperty("log4j.appender.R2.layout",                   "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.R2.layout.ConversionPattern", "%m%n");

	    p.setProperty("log4j.appender.R3",                          "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.R3.Threshold",                "INFO");
	    p.setProperty("log4j.appender.R3.File",                     logPathFileName + "-3-info.log");
	    p.setProperty("log4j.appender.R3.MaxFileSize",              "2MB");
	    p.setProperty("log4j.appender.R3.MaxBackupIndex",           "3");
	    p.setProperty("log4j.appender.R3.layout",                   "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.R3.layout.ConversionPattern", "%m%n");

	    p.setProperty("log4j.appender.R4",                          "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.R4.Threshold",                "WARN");
	    p.setProperty("log4j.appender.R4.File",                     logPathFileName + "-4-warn.log");
	    p.setProperty("log4j.appender.R4.MaxFileSize",              "256KB");
	    p.setProperty("log4j.appender.R4.MaxBackupIndex",           "3");
	    p.setProperty("log4j.appender.R4.layout",                   "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.R4.layout.ConversionPattern", "%m%n");

	    p.setProperty("log4j.appender.R5",                          "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.R5.Threshold",                "ERROR");
	    p.setProperty("log4j.appender.R5.File",                     logPathFileName + "-5-error.log");
	    p.setProperty("log4j.appender.R5.MaxFileSize",              "256KB");
	    p.setProperty("log4j.appender.R5.MaxBackupIndex",           "3");
	    p.setProperty("log4j.appender.R5.layout",                   "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.R5.layout.ConversionPattern", "%m%n");

	    p.setProperty("log4j.appender.R6",                          "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.R6.Threshold",                "FATAL");
	    p.setProperty("log4j.appender.R6.File",                     logPathFileName + "-6-fatal.log");
	    p.setProperty("log4j.appender.R6.MaxFileSize",              "256KB");
	    p.setProperty("log4j.appender.R6.MaxBackupIndex",           "3");
	    p.setProperty("log4j.appender.R6.layout",                   "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.R6.layout.ConversionPattern", "%m%n");

	    // HTTPD logging
	    p.setProperty("log4j.logger.water.api.RequestServer",       "TRACE, HTTPD");

	    p.setProperty("log4j.appender.HTTPD",                       "org.apache.log4j.RollingFileAppender");
	    p.setProperty("log4j.appender.HTTPD.Threshold",             "TRACE");
	    p.setProperty("log4j.appender.HTTPD.File",                  logPathFileName + "-httpd.log");
	    p.setProperty("log4j.appender.HTTPD.MaxFileSize",           "1MB");
	    p.setProperty("log4j.appender.HTTPD.MaxBackupIndex",        "3");
	    p.setProperty("log4j.appender.HTTPD.layout",                "org.apache.log4j.PatternLayout");
	    p.setProperty("log4j.appender.HTTPD.layout.ConversionPattern", "%m%n");

	    // Turn down the logging for some class hierarchies.
	    /* Not yet integrated with any of these... but leave for now... */
	    p.setProperty("log4j.logger.org.apache.http",               "WARN");
	    p.setProperty("log4j.logger.com.amazonaws",                 "WARN");
	    p.setProperty("log4j.logger.org.apache.hadoop",             "WARN");
	    p.setProperty("log4j.logger.org.jets3t.service",            "WARN");

	    // See the following document for information about the pattern layout.
	    // http://logging.apache.org/log4j/1.2/apidocs/org/apache/log4j/PatternLayout.html
	    //
	    //  Uncomment this line to find the source of unwanted messages.
		//     p.setProperty("log4j.appender.R1.layout.ConversionPattern", "%p %C %m%n");
	}
	
	private static org.apache.log4j.Logger createLog4jLogger(String logDirParent) {
		synchronized (com.clust4j.log.Log.class) {
			// H2O API is synchronized here... this is not:
			if(null != _logger)
				return _logger;
			
			String l4jprops = System.getProperty("log4j.properties");
			if(null != l4jprops)
				PropertyConfigurator.configure(l4jprops);
			
			else {
				java.util.Properties p = new java.util.Properties();
				setLog4jProperties(logDirParent, p);
				PropertyConfigurator.configure(p);
			}
		}
		
		return _logger = LogManager.getLogger(Log.class.getName());
	}
	
	public static void setLogLevel(int log_level) throws IllegalArgumentException {
		Level l;
		
		switch(log_level) {
			case 1: l = Level.TRACE; break;
			case 2: l = Level.DEBUG; break;
			case 3: l = Level.INFO;	 break;
			case 4: l = Level.WARN;  break;
			case 5: l = Level.ERROR; break;
			case 6: l = Level.FATAL; break;
			default:
				throw new IllegalArgumentException("Illegal log level: "+ log_level);
		}
		
		_logger.setLevel(l);
		String inf = "Set log level to " + l;
		System.out.println(inf);
		_logger.info(inf);
	}
	
	
	/**
	 * Volatile in H2O API, not here....
	 */
	static volatile boolean loggerCreateWasCalled = false;
	static private Object startupLogEventsLock = new Object();
	static volatile private ArrayList<LogEvent> startupLogEvents = new ArrayList<LogEvent>();
	
	
	protected static void log0(org.apache.log4j.Logger l4j, LogEvent e) {
		String s = e.toString();
		
		if(e.type == Type.FATAL)
			l4j.fatal(s);
		else if(e.type == Type.ERROR)
			l4j.error(s);
		else if(e.type == Type.WARN)
			l4j.warn(s);
		else if(e.type == Type.INFO)
			l4j.info(s);
		else if(e.type == Type.DEBUG)
			l4j.debug(s);
		else if(e.type == Type.TRACE)
			l4j.trace(s);
		
		else l4j.error(s); // DEFAULT ERROR IF WE CAN'T FIGURE OUT LEVEL...
	}
	
	
	
	private static void write0(final LogEvent e, final boolean printOnOut) {
		org.apache.log4j.Logger l4j = getLog4jLogger();
		
		// If we don't have a logger yet, and we haven't created one, build one...
		//synchronized(Event.class) {
			if((null == l4j) && !loggerCreateWasCalled) {
				File dir;
				
				final URI root = com.clust4j.log.LogProperties.getRoot();
				boolean windowsPath = root.toString().matches("^[a-zA-Z]:.*");
				
				if(windowsPath)
					dir = new File(root.toString());
				else if(root.getScheme() == null || "file".equals(root.getScheme()))
					dir = new File(root.getPath());
				else 
					dir = new File(com.clust4j.log.LogProperties.DEFAULT_ROOT());
				
				loggerCreateWasCalled = true;
				l4j = createLog4jLogger(dir.toString());
				info(Algo.CLUST4J, "Logging at "+dir.toString());
			}
		//}
		
		
		// Log if we can, or buffer
		if(null == l4j) {
			e.toString();
			
			
			synchronized(startupLogEventsLock) {
				if(startupLogEvents != null)
					startupLogEvents.add(e);	
				else {
					// Startup race condition here to be aware of
				}
			}
			
		} else {
			if(startupLogEvents != null) {
				synchronized(startupLogEventsLock) {
					for(int i = 0; i < startupLogEvents.size(); i++) {
						LogEvent bufferedEvent = startupLogEvents.get(i);
						log0(l4j, bufferedEvent);
					}
					
					startupLogEvents = null;
				}
			}
			
			log0(l4j, e);
		}
		
		
		if(printOnOut || printAll)
			unwrap(System.out, e.toShortString());
		e.printMe = false;
	}
	
	
	
	
	
	
	static public void err(Algo t, String msg) {
		LogEvent e = LogEvent.make(t, Type.ERROR, null, msg);
		write(e, true);
	}
	
	static public <T extends Throwable> T warn(Algo t, String msg, T exception) {
		LogEvent e = LogEvent.make(t, Type.WARN, exception, msg);
		write(e, true);
		return exception;
	}
	
	static public Throwable warn(Algo t, String msg) {
		return warn(t, msg, null);
	}
	
	static public void info(Algo t, Object... obj) {
		LogEvent e = LogEvent.make(t, Type.INFO, null, obj);
		write(e, true);
	}
	
	static public void info(Object... objects) {
		info(Algo.CLUST4J, objects);
	}
	
	static public void debug(Algo t, Object... objects) {
		if(flag(t) == false)
			return;
		LogEvent e = LogEvent.make(t, Type.DEBUG, null, objects);
		write(e, false);
	}
	
	static public void trace(Object... objects) {
		if(flag(Algo.CLUST4J) == false)
			return;
		LogEvent e = LogEvent.make(Algo.CLUST4J, Type.TRACE, null, objects);
		write(e, false);
	}
	
	
	
	
	
	
	
	public static final Type[] TYPES = Type.values();
	public static final Algo[] ALGOS = Algo.values();
	private static final String lineSep = System.getProperty("line.separator");
	static String LOG_DIR = null; // Want to log to console...
	static final Timer time = new LogTimer();
	
	
	
	private static boolean printAll;
	static {
		String pa = System.getProperty("log.printAll");
		printAll = (pa != null && pa.equals("true"));
		
		/* Default, log everything for all algos */
		for(Algo a: ALGOS)
			setFlag(a);
		
		/* Unflag those which are explicitly NOFLAG */
		for(Algo s : ALGOS) {
			String str = System.getProperty("log."+s);
			if (null == str) continue;
			if (str.equals("false")) unsetFlag(s); else setFlag(s);
	    }
	}
	
	
	/** Check if a subsystem will print debug message to the LOG file */
	public static boolean flag(Algo t) { return t._enable || printAll; }
	/** Set the debug flag. */
	public static void setFlag(Algo t) { t._enable = true; }
	/** Unset the debug flag. */
	public static void unsetFlag(Algo t) { t._enable = false; }
	
	
	
	public static void unwrap(PrintStream stream, String s) {
		if(stream instanceof LogWrapper)
			((LogWrapper)stream).printlnParent(s);
		else stream.println(s);
	}
}
