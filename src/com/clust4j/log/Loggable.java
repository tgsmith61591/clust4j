package com.clust4j.log;

public interface Loggable {
	public void error(String msg);
	public void warn(String msg);
	public void info(String msg);
	public void trace(String msg);
	public void debug(String msg);
	public void sayBye(LogTimer timer);
	public com.clust4j.log.Log.Tag.Algo getLoggerTag();
	public boolean hasWarnings();
}
