# CHS higher=better
# MSC higher=better
# DBS  lower=better
# SDbw lower=better

def metric():
  if args.nometric == 1: return
  
  MSG1("metric")
  from sklearn import metrics
  CHS = metrics.calinski_harabasz_score(vectors,idx)
  MSG2(f"CHS^={CHS:.2f}")

  MSC = metrics.silhouette_score(vectors,idx)
  MSG2(f"MSC^={MSC:.3f}")

  DBS = metrics.davies_bouldin_score(vectors,idx)
  MSG2(f"DBS={DBS:.3f}")

  import validclust
  pwdist = metrics.pairwise_distances(vectors)
  COP = validclust.cop(vectors,pwdist,idx)
  MSG2(f"COP={COP:.3f}")

  from s_dbw import S_Dbw
  SDbw = S_Dbw(vectors,idx,centers_id=None,method='Tong',alg_noise='bind',centr='mean',nearest_centr=True,metric='euclidean')
  MSG2(f"SDbw={SDbw:.3f}")

  MSG3("(^= means higher better)")

