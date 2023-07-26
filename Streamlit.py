import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans

df = pd.read_csv('Mall_Customers.csv')

# rename kolom
df.rename(index=str, columns={
    'Annual Income (k$)' : 'Income',
    'Spending Score (1-100)' : 'Score'
}, inplace = True)

# Drop colomn yang tidak digunakan
X = df.drop(['CustomerID', 'Gender'], axis=1)
# menghapus colomn customer dan gender 

st.header("isi dataset clear : ")

st.write(X)
# ini digunakan untuk menampilkan di streamlit nya

# untuk menentukan berapa cluster yang akan digunakan atau di pakai
# paling afdol tuh menggunakan elbow untuk menentukannya 
# dan memilih elbow yang berada di tengah2 atau mencari nilai tengah elbow
clusters=[]
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)
fig, ax = plt.subplots(figsize=(12,8))
# untuk figure / gambar nya
sns.lineplot(x=list(range(1,11)),y=clusters, ax=ax)
# untuk garisnya
ax.set_title('mencari elbow')
ax.set_xlabel('clusters')
ax.set_ylabel('inertia')
# digunakan untuk memberi panah pada elbow
ax.annotate('Possible elbow point', xy=(3, 140000), xytext=(3, 50000), xycoords='data', arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))
ax.annotate('Possible elbow point', xy=(5, 80000), xytext=(5, 150000), xycoords='data', arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2))

# menampilkan di streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()


# menampilkan sidebar
st.sidebar.subheader("Nilai jumlah K ")
clust = st.sidebar.slider(" Pilih Jumlah Cluster : ", 2, 10, 3, 1) #range 2-10, start, step

# fungsi/tahapan clustering nya
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    plt.figure(figsize=(10,8))
    sns.scatterplot(x=X['Income'], y=X['Score'], hue=X['Labels'], markers=True, size=X['Labels'], palette=sns.color_palette('hls', n_clust))

    for label in X['Labels']:
        plt.annotate(label,
            (X[X['Labels']==label]['Income'].mean(),
            X[X['Labels']==label]['Score'].mean()),
            horizontalalignment = 'center',
            verticalalignment = 'center',
            size = 20, weight='bold',
            color ='black')
    
    st.header('cluster plot')
    st.pyplot()

    st.header('cluster berdasarkan table')
    st.write(X)


k_means(clust)


