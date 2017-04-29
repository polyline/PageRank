
package org.apache.hadoop.examples;

import java.io.IOException;
import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import com.google.common.collect.Iterables;
import org.apache.hadoop.conf.Configuration;


public class PageRank {
	
	private static final String OUTPUT_PATH = "intermediate_output";
	private static final double BETA = 0.8;
	private static final int sizeOfTotalNodes = 10880;
	private static final int iterationTimes = 20;
	
	
    public static class PageRankMapper
        extends Mapper<Object, Text, Text, Text>{
		
		private Text outputKey = new Text();
		private Text outputValue = new Text();
		
		public void map(Object key, Text value, Context context
               ) throws IOException, InterruptedException {

			String data = value.toString();
			String[] args = data.split("	");
			
			outputKey.set(args[0]);
			outputValue.set(args[1]);
			context.write(outputKey, outputValue);
			
			outputKey.set("C");
			outputValue.set(args[0]);
			context.write(outputKey, outputValue);
			
			outputKey.set("C");
			outputValue.set(args[1]);
			context.write(outputKey, outputValue);
    }
}

public static class PageRankReducer
        extends Reducer<Text,Text,Text,Text> {
		private MultipleOutputs<Text, Text> mos;

	@Override
	protected void setup(Context context) {
		mos = new MultipleOutputs(context);
	}
	
    public void reduce(Text key, Iterable<Text> values,
                        Context context
                        ) throws IOException, InterruptedException {

		if(key.toString().equals("C")){
			//sto is used to record whether the node exists or not
			boolean [] sto = new boolean[sizeOfTotalNodes];
			for(int i=0; i<sizeOfTotalNodes; i++){
				sto[i] = false;
			}
			for(Text value: values){
				int idx = Integer.parseInt(value.toString());
				if(sto[idx] == false){
					sto[idx] = true;
				}
			}
			int sum = 0;
			for(int i=0; i<sizeOfTotalNodes; i++){
				if(sto[i]){
					sum++;
				}
			}
			Configuration conf = context.getConfiguration();
			conf.set("ExistNodes", Integer.toString(sum));
			double iniVal = (double) 1 / sum * (1-BETA);
			double iniValOfR = (double) 1 / sum ;
			String strVal = Double.toString(iniVal);
			String strValOfR = Double.toString(iniValOfR);
			for(int i=0; i<sizeOfTotalNodes; i++){
				if(sto[i]){
					mos.write("MN", NullWritable.get(), new Text("N,"+Integer.toString(i)+","+strVal));
					mos.write("R", NullWritable.get(), new Text("R,"+Integer.toString(i)+","+strValOfR));
				}
			}
		}else{
			
			Vector v = new Vector(); 
			String result = "";
			int size = 0;
			int sum = 0;
			for(Text value: values){
				v.add("M,"+value.toString()+"," + key.toString() + ",");
				result = result + value.toString() + ",";
				sum++;
				size++;
			}
			double val = (double) 1 / size * BETA;
			String strVal = Double.toString(val);
			
			for(int i=0; i<v.size(); i++){
				mos.write("MN", NullWritable.get(), new Text(v.get(i) + strVal));
				
			}
			mos.write("EN", NullWritable.get(), new Text(key.toString() + "," + Integer.toString(sum) + "," + result));
		}
		
    }
	@Override
	protected void cleanup(Context context) throws IOException, InterruptedException {
		mos.close();
	}
}

public static class MultipliMapper extends Mapper<Object, Text, Text, Text>{
	
	//initialize outputKey and outputValue. They are Text
	//initialize them outside, so it can reduce the usage of memory
	Text outputKey = new Text();
	Text outputValue = new Text();
	
	public void map(Object key, Text value, Context context
               ) throws IOException, InterruptedException {
		/**
			Get matrix M and R and N
			Emit pair in this form:
				Key: i
				Value: ({M/N}, j, value)
				or Value: (R, value)
		**/
		String data = value.toString();
		/**
			data-form
			M: M,Row,Col,Val
			N: M,Row,Val
			R: R,Row,Val
		**/
		String[] args = data.split(",");
		if(args[0].equals("M")){
			outputKey.set(args[1]);
			//(M,j,value), M=args[0], j=args[2], value=args[3]
			outputValue.set(args[0]+","+args[2]+","+args[3]);
			//write to context
			context.write(outputKey, outputValue);
		}
		else if(args[0].equals("R")){
			//R,Node,Val,Node,Number,To1,To2,...
			int num = Integer.parseInt(args[4]);
			if(num != -1){
				//if num equals to -1, then the nodes outlines 0 nodes
				for(int i=0; i<num; i++){
					//real position of the ToNode
					int pos = i+5;
					outputKey.set(args[pos]);
					outputValue.set(args[0]+","+args[1]+","+args[2]);
					context.write(outputKey, outputValue);
				}
			}
		}
		else{
			outputKey.set(args[1]);
			outputValue.set(args[0]+","+args[2]);
			//write to context
			context.write(outputKey, outputValue);
		}
	}
}

public static class MultipliReducer extends Reducer<Text,Text,Text,Text> 
{
		public void reduce(Text key, Iterable<Text> values,
                        Context context
                        ) throws IOException, InterruptedException {
        /**
			data-form
			   Key				Value
			M: Row				M,j,Val
			N: Row				N,Val
			R: Row				R,j,Val
		**/
		HashMap<Integer, Double> hashMapM = new HashMap<Integer, Double>();
		HashMap<Integer, Double> hashMapR = new HashMap<Integer, Double>();
		//
		double nVal = 0.0;
		for(Text value: values){
			String[] args = value.toString().split(",");
			//{M/N}, j, value
			if(args[0].equals("M")){
				//put value in j bucket
				hashMapM.put(Integer.parseInt(args[1]), Double.parseDouble(args[2]));
			}else if(args[0].equals("N")){
				nVal = Double.parseDouble(args[1]);
			}else{
				//put value in j bucket
				hashMapR.put(Integer.parseInt(args[1]), Double.parseDouble(args[2]));
			}
		}
		double result = 0.0;
		for(int k=0; k<sizeOfTotalNodes; k++){
			double m_ij = hashMapM.containsKey(k) ? hashMapM.get(k) : 0;
			double r_i = hashMapR.containsKey(k) ? hashMapR.get(k) : 0;
			result += m_ij * r_i;
		}
		//the rule to handle the non-exist nodes is to ignore it and don't output it
		result += nVal;
		//if the key not exist, then the value of it will be zero
		context.write(null, new Text("R," + key.toString() + "," + Double.toString(result)));
	}
}

public static class RenormalMapper extends Mapper<Object, Text, Text, Text>{
	
	public void map(Object key, Text value, Context context
               ) throws IOException, InterruptedException {
		String data = value.toString();
		String[] args = data.split(",");
		Text outputKey = new Text();
		Text outputValue = new Text();
		outputKey.set("Re");
		outputValue.set(args[1] +","+ args[2]);
		context.write(outputKey, outputValue);
	}
}

public static class RenormalReducer extends Reducer<Text,Text,Text,Text> 
{
		public void reduce(Text key, Iterable<Text> values,
                        Context context
                        ) throws IOException, InterruptedException {
			Vector v = new Vector(); 
			HashMap<Integer, Double> hashMapRe = new HashMap<Integer, Double>();
			double sum = 0.0;
			int size = 0;
			for(Text value: values){
			String [] args = value.toString().split(",");
				double val = Double.parseDouble(args[1]);
				hashMapRe.put(Integer.parseInt(args[0]),val);
				sum += val;
				size++;
				
			}
			double adjVal = (double)(1 - sum) / size; 
			for(int i=0; i<sizeOfTotalNodes; i++){
				if(hashMapRe.containsKey(i)){
					double newVal = (double)hashMapRe.get(i);
					 newVal += adjVal;
					 //only write R if it exists
					 context.write(null, new Text("R,"+ Integer.toString(i) +","+ Double.toString(newVal)));
				}
				
			}
		}
}

public static void main(String[] args) throws Exception {

	Configuration conf = new Configuration();
    String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
        System.err.println("Usage: matrix multiplication <in> <out>");
        System.exit(2);
    }
	
	conf.set("path", otherArgs[1] + "/0");
	/**
	--First Job--
	calculate the M matrix
	calculate the N vector(determine the existed points and assign 1/N as its value)
	initialize the r vector with 1/N
	**/
    Job job = new Job(conf, "PageRank");
    job.setJarByClass(PageRank.class);
    job.setMapperClass(PageRankMapper.class);
    job.setReducerClass(PageRankReducer.class);

    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Text.class);

    job.setInputFormatClass(TextInputFormat.class);
    job.setOutputFormatClass(TextOutputFormat.class);

    FileInputFormat.addInputPath(job, new Path(otherArgs[0]));
    FileOutputFormat.setOutputPath(job, new Path(otherArgs[1] + "/0"));

	MultipleOutputs.addNamedOutput(job, "MN", TextOutputFormat.class, NullWritable.class, Text.class);
	MultipleOutputs.addNamedOutput(job, "R", TextOutputFormat.class, NullWritable.class, Text.class);
	MultipleOutputs.addNamedOutput(job, "EN", TextOutputFormat.class, NullWritable.class, Text.class);
    //System.exit(job.waitForCompletion(true) ? 0 : 1);}
	job.waitForCompletion(true) ;
	/**
	Iteration
	Run N times
	**/
	for(int i=0; i<iterationTimes; i++){
		/**
		Read the EN and R 
		Combine them
		**/
		Vector En =  new Vector();
		HashMap<Integer, String> hashMapEn = new HashMap<Integer, String>();
		HashMap<Integer, String> hashMapR = new HashMap<Integer, String>();

		FileSystem fs = FileSystem.get(conf);
		
		Path ENPath= new Path( otherArgs[1] + "/0" + "/EN-r-00000");
		InputStream is = fs.open(ENPath);
        BufferedReader br=new BufferedReader(new InputStreamReader(is));
        String line = br.readLine();
        while (line != null){

				/**
				FromNode,Number,ToNode1,ToNode2
				**/
				String [] sep = line.split(",");
				int node = Integer.parseInt(sep[0]);
				hashMapEn.put(node, line);              
				line = br.readLine();
       }
	   	Path RPath= new Path( otherArgs[1] + "/" + Integer.toString(i) + "/R-r-00000");
		InputStream is2 = fs.open(RPath);
		BufferedReader br2 =new BufferedReader(new InputStreamReader(is2));
        line = br2.readLine();
        while (line != null){
				/**
				R,node,val
				**/
				String [] sep = line.split(",");
				int node = Integer.parseInt(sep[1]);
				hashMapR.put(node, line);       
				line = br2.readLine();
       }
	   
	   Path NRpath = new Path( otherArgs[1] + "/" + Integer.toString(i) +"/NR-r-00000");
	   BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(fs.create(NRpath, true), "UTF-8"));
	   for(int k=0; k<sizeOfTotalNodes; k++){
			//R,Node,Val,Node,Number,To1,To2,...
			String en = hashMapEn.containsKey(k) ? hashMapEn.get(k) : "-1,-1";
			String r = "";
			if(hashMapR.containsKey(k)){
				r = hashMapR.get(k);
				bw.write(r + "," + en + "\n");
			}
		}
		bw.flush();
		bw.close();
		/**
		--Second Job--
		Multiplication
		R' = M * R + N
		**/
		Job job2 = new Job(conf, "Multiplication");
		job2.setJarByClass(PageRank.class);
		job2.setMapperClass(MultipliMapper.class);
		job2.setReducerClass(MultipliReducer.class);

		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);

		job2.setInputFormatClass(TextInputFormat.class);
		job2.setOutputFormatClass(TextOutputFormat.class);
	
		MultipleInputs.addInputPath(job2 , new Path(otherArgs[1] + "/0" + "/MN-r-00000*"), TextInputFormat.class, MultipliMapper.class);
		MultipleInputs.addInputPath(job2 , new Path(otherArgs[1] + "/" + Integer.toString(i) + "/NR-r-00000*"), TextInputFormat.class, MultipliMapper.class);
		
		//change the basename of the output
		job2.getConfiguration().set("mapreduce.output.basename", "tmp");

		/**
			Delete the "temp" directory, if it exists.
			Since we need to reuse this name
		**/
		Path temp = new Path(otherArgs[1] + "/temp");
		FileSystem hdfs = FileSystem.get(conf);
		// delete existing directory
		if (hdfs.exists(temp)) {
			hdfs.delete(temp, true);
		}
		FileOutputFormat.setOutputPath(job2, new Path(otherArgs[1]+"/temp"));
		job2.waitForCompletion(true);
		/**
		--Third Job--
		Renormalize the PageRank
		**/
		Job job3 = new Job(conf, "Renormalization");
		job3.setJarByClass(PageRank.class);
		job3.setMapperClass(RenormalMapper.class);
		job3.setReducerClass(RenormalReducer.class);

		job3.setOutputKeyClass(Text.class);
		job3.setOutputValueClass(Text.class);

		job3.setInputFormatClass(TextInputFormat.class);
		job3.setOutputFormatClass(TextOutputFormat.class);
	
		//MultipleInputs.addInputPath(job3 , new Path(otherArgs[1] + "/MN-r-00000*"), TextInputFormat.class, MultipliMapper.class);
		//MultipleInputs.addInputPath(job3 , new Path(otherArgs[1] + "/R-r-00000*"), TextInputFormat.class, MultipliMapper.class);
		
		job3.getConfiguration().set("mapreduce.output.basename", "R");
		FileInputFormat.addInputPath(job3, new Path(otherArgs[1] + "/temp/tmp-r-00000*"));
		FileOutputFormat.setOutputPath(job3, new Path(otherArgs[1] + "/" + Integer.toString(i+1)));
		MultipleOutputs.addNamedOutput(job3, "newr", TextOutputFormat.class, NullWritable.class, Text.class);
    
		job3.waitForCompletion(true);
		
		
		
	}
	/**
	sorting and give the top ten
	**/
		Path UnsortedPath= new Path( otherArgs[1] + "/" + Integer.toString(iterationTimes) + "/R-r-00000");
        FileSystem fs = FileSystem.get(conf);
		BufferedReader rbr= new BufferedReader(new InputStreamReader(fs.open(UnsortedPath)));
        String line = rbr.readLine();
		HashMap<String, Double> hm = new HashMap<String, Double>();
        while (line != null){
				/**
				R,row,val
				**/
				String [] sep = line.split(",");
				hm.put(sep[1], Double.parseDouble(sep[2]));              
				line = rbr.readLine();
       }
        Set<Entry<String, Double>> set = hm.entrySet();
        List<Entry<String, Double>> list = new ArrayList<Entry<String, Double>>(
                set);
				
        Collections.sort(list, new Comparator<Map.Entry<String, Double>>() {
            public int compare(Map.Entry<String, Double> o1,
                    Map.Entry<String, Double> o2) {
                return o2.getValue().compareTo(o1.getValue());
            }
        });
		Path Sortedpath = new Path( otherArgs[1] + "/" +"/topTen");
		BufferedWriter sbw = new BufferedWriter(new OutputStreamWriter(fs.create(Sortedpath, true), "UTF-8"));
		int topTen = 0;
		for (Entry<String, Double> entry : list) {
			if(topTen >= 10) break;
			sbw.write(entry.getKey()+","+ Double.toString(entry.getValue()) + "\n");
			topTen++;
        }
		sbw.flush();
		sbw.close();
	}
}

