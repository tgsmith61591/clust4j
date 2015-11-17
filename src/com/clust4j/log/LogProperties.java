package com.clust4j.log;

import java.net.URI;
import java.net.URISyntaxException;

public class LogProperties {
	public static String DEFAULT_ROOT() {
		String usr = System.getProperty("user.name");
		if(null == usr)
			usr = "";
		
		String usr2 = usr.replaceAll(" ", "_");
		if(usr2.length() == 0)
			usr2 = "unknown";
		
		return "/tmp/clust4j-" + usr2;
	}
	
	
	static String root = DEFAULT_ROOT();
	private static URI ROOT;
	
	static {
		try {
			ROOT = new URI(root);
		} catch(URISyntaxException e) {
			throw new RuntimeException("Invalid root: " + root + ", " + e.getMessage());
		}
	}
	
	public static URI getRoot() {
		return ROOT;
	}
}
