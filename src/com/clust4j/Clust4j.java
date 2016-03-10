package com.clust4j;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**
 * The absolute super type for all clust4j objects (models and datasets)
 * that should be able to commonly serialize their data.
 * @author Taylor G Smith
 */
public abstract class Clust4j implements java.io.Serializable {
	private static final long serialVersionUID = -4522135376738501625L;

	
	/**
	 * Load a model from a FileInputStream
	 * @param fos
	 * @return
	 * @throws IOException
	 * @throws ClassNotFoundException
	 */
	public static Clust4j loadObject(final FileInputStream fis) throws IOException, ClassNotFoundException {
		ObjectInputStream in = null;
		Clust4j bm = null;
			
		try {
			in = new ObjectInputStream(fis);
	        bm = (Clust4j) in.readObject();
		} finally {
			try {
				in.close();
			} catch(NullPointerException n) {
				// only happens if improperly initialized...
			}
	        
	        fis.close();
		}
        
        return bm;
	}
	
	/**
	 * Save a model to FileOutputStream
	 * @param fos
	 * @throws IOException
	 */
	public void saveObject(final FileOutputStream fos) throws IOException {
		ObjectOutputStream out = null;
		
		try {
			out = new ObjectOutputStream(fos);
			out.writeObject(this);
		} finally {
			try {
				out.close();
			} catch(NullPointerException n) {
				// only happens if improperly initialized...
			}
				
			fos.close();
		}
	}
}
