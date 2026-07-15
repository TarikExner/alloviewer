import { useCallback, useRef, useState } from "react";
import {
  createJob,
  deleteJob,
  type JobType,
} from "../api";

export function useJobSession(jobType: JobType) {
  const [jobId, setJobId] = useState<string | null>(null);

  const jobIdRef = useRef<string | null>(null);
  const creationRef = useRef<Promise<string> | null>(null);

  const ensureJobId = useCallback(async (): Promise<string> => {
    if (jobIdRef.current) {
      return jobIdRef.current;
    }

    if (creationRef.current) {
      return creationRef.current;
    }

    const pending = createJob(jobType)
      .then((job) => {
        jobIdRef.current = job.job_id;
        setJobId(job.job_id);
        return job.job_id;
      })
      .finally(() => {
        creationRef.current = null;
      });

    creationRef.current = pending;
    return pending;
  }, [jobType]);

  const resetJob = useCallback(
    async (deleteRemote = false): Promise<void> => {
      const current = jobIdRef.current;

      jobIdRef.current = null;
      creationRef.current = null;
      setJobId(null);

      if (deleteRemote && current) {
        try {
          await deleteJob(current);
        } catch {
          // Cleanup task will remove abandoned jobs.
        }
      }
    },
    []
  );

  return {
    jobId,
    ensureJobId,
    resetJob,
  };
}
