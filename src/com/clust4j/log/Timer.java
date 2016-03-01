package com.clust4j.log;

public interface Timer {
	public long time();
	public long nanos();
	public String startAsString();
	public String startAsShortString();
	public String nowAsString();
	public String nowAsShortString();
}
